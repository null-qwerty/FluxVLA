#!/usr/bin/env python
# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert EgoDex episodes into LLaVA-format VLM SFT annotations.

EgoDex (https://github.com/apple/ml-egodex) is first-person RGB video of human
hand manipulation with MANO hand poses and camera poses. It has **no robot
joint actions, no robot states and no language instructions**, so it can only
feed the VLM branch of FluxVLA mixed training. This tool samples frames from
each episode video, writes them as JPEG files and emits one LLaVA-format JSONL
row per sampled frame with a deterministic two-turn conversation.

Expected input layout (EgoDex README)::

    <root>/
        train/part1/<task>/<episode>.hdf5
        train/part1/<task>/<episode>.mp4
        ...
        extra/<task>/<episode>.hdf5
        extra/<task>/<episode>.mp4

Any `*.hdf5` with a sibling `*.mp4` is converted, so `test/`, `val/`
or a flat directory tree also work. The task name is the HDF5 parent directory
name; `--tasks` filters on it.

HDF5 fields used here (all optional; a missing key only reduces caption
detail, it never aborts a conversion):

`camera/intrinsic`
    `(3, 3)` float pinhole intrinsic matrix `[[fx, 0, cx], [0, fy, cy],
    [0, 0, 1]]`. Per-frame `(T, 3, 3)` stacks are accepted too. Only its
    presence/shape is recorded; frames are not undistorted.
`transforms/camera`
    `(T, 4, 4)` float camera-to-world transform per video frame. `T`
    bounds the sampled frame indices.
`leftHand/transforms` / `rightHand/transforms`
    `(T, 4, 4)` MANO wrist pose per frame. Only used to infer whether a hand
    is moving when no confidence is available.
`leftHand/confidences` / `rightHand/confidences`
    `(T,)` per-frame hand tracking confidence in `[0, 1]`. Only used to
    infer which hand is active in a caption.

Run `--inspect <file.hdf5>` to print the actual EgoDex schema of a local
episode before converting, and read the EgoDex README for the authoritative
field definitions.

Output::

    <out>/ego.jsonl               # one record per sampled frame
    <out>/images/<task>/<ep>.jpg  # relative paths inside the JSONL
    <out>/convert_summary.json    # counts, config, per-task statistics

Each JSONL row::

    {"id": ..., "image": "images/<task>/<episode>_f<frame>.jpg",
     "conversations": [{"from": "human", "value": "<image>\n<question 1>"},
                       {"from": "gpt", "value": "<answer 1>"},
                       {"from": "human", "value": "<question 2>"},
                       {"from": "gpt", "value": "<answer 2>"}],
     "dataset": "egodex", "task": ..., "episode": ..., "frame_index": ...,
     "timestamp_s": ..., "source_video": ..., "source_hdf5": ...}

The two-turn schema is the one Pi05 mixed training expects: the image must be
referenced from the first user turn, later turns are text only.

Example::

    python tools/egodex_to_vlm_jsonl.py \
        --root /mnt/data/oss/limx_mani_data/raw_data/EgoDex_test \
        --out /mnt/data/oss/users/neri/data/egodex_vlm \
        --tasks pour,open_drawer --max-episodes 50 --fps-sample 1

This tool deliberately does not read or write robot actions. Do not add EgoDex
rows to a VLA dataloader: a VLM batch has no states/actions and the VLA loss is
action flow MSE only.
"""

import argparse
import concurrent.futures
import fnmatch
import glob
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import zlib

LOGGER = logging.getLogger('egodex_to_vlm_jsonl')

_CAMERA_INTRINSIC = 'camera/intrinsic'
_CAMERA_TRANSFORMS = 'transforms/camera'
_HANDS = ('leftHand', 'rightHand')
_ACTIVE_CONFIDENCE = 0.5

_QUESTIONS_ONE = (
    'What is the person doing in this egocentric frame?',
    'Describe the manipulation happening in this first-person view.',
    'What task is the person performing in this first-person video?',
    'What action is the person carrying out in this egocentric image?',
)
_QUESTIONS_TWO = (
    'Which hands are involved, and what are they doing?',
    'Are both hands used for this manipulation? Explain.',
    'What are the hands doing in this first-person view?',
    'How is the person using their hands in this frame?',
)
_HAND_PHRASES = {
    'both':
    ('Both hands are visible in the frame and are actively manipulating '
     'objects.'),
    'left':
    ('The left hand is the active manipulator, while the right hand assists '
     'or stays out of the way.'),
    'right':
    ('The right hand is the active manipulator, while the left hand assists '
     'or stays out of the way.'),
    'none': ('Neither hand is clearly manipulating an object in this frame.'),
    None: ('The hands are interacting with objects on the work surface.'),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--root', help='EgoDex root directory containing train/ and extra/.')
    parser.add_argument(
        '--out', help='Output directory for ego.jsonl and extracted images/.')
    parser.add_argument(
        '--tasks',
        default=None,
        help='Comma-separated task directory names (fnmatch patterns '
        'allowed). Default: every discovered task.')
    parser.add_argument(
        '--max-episodes',
        type=int,
        default=0,
        help='Maximum number of converted episodes per task. 0 means all.')
    parser.add_argument(
        '--fps-sample',
        type=float,
        default=1.0,
        help='Frames sampled per second of video. Default 1.0.')
    parser.add_argument(
        '--max-frames-per-episode',
        type=int,
        default=8,
        help='Upper bound on sampled frames for one episode. Default 8.')
    parser.add_argument(
        '--image-max-side',
        type=int,
        default=448,
        help='Downscale so that the long side is at most this many pixels. '
        '0 keeps the original resolution. Default 448.')
    parser.add_argument(
        '--jpeg-quality',
        type=int,
        default=90,
        help='JPEG export quality. Default 90.')
    parser.add_argument(
        '--captioner',
        choices=('template', 'vlm'),
        default='template',
        help='Caption source. "template" builds deterministic task-name '
        'captions; "vlm" requires --captioner-module.')
    parser.add_argument(
        '--captioner-module',
        default=None,
        help='Dotted module:attribute providing the --captioner vlm '
        'callable.')
    parser.add_argument(
        '--captioner-model',
        default=None,
        help='Optional model path/name forwarded to the vlm captioner.')
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Parallel episode conversions. Default 4.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace an existing ego.jsonl/images directory.')
    parser.add_argument(
        '--inspect',
        default=None,
        metavar='HDF5',
        help='Print the HDF5 schema of one EgoDex episode and exit.')
    return parser.parse_args(argv)


def humanize_task(task):
    """Turn an EgoDex task directory name into caption-ready text."""
    return re.sub(r'[_\s]+', ' ', task).strip().lower()


def _natural_key(path):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r'(\d+)', path)
    ]


def discover_episodes(root):
    """Return {task: [episode dicts]} for every HDF5 with a sibling video."""
    episodes = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(dirnames)
        for name in sorted(filenames):
            if not name.endswith('.hdf5'):
                continue
            hdf5_path = os.path.join(dirpath, name)
            video_path = hdf5_path[:-len('.hdf5')] + '.mp4'
            if not os.path.isfile(video_path):
                LOGGER.warning('Skipping %s: no sibling .mp4', hdf5_path)
                continue
            relative = os.path.relpath(hdf5_path, root)
            parts = relative.split(os.sep)
            task = parts[-2] if len(parts) > 1 else 'unknown'
            split = parts[0] if len(parts) > 2 else ''
            episode_id = os.path.splitext(relative)[0].replace(os.sep, '/')
            episodes.setdefault(task, []).append(
                dict(
                    task=task,
                    split=split,
                    episode_id=episode_id,
                    hdf5=hdf5_path,
                    video=video_path))
    for task in episodes:
        episodes[task].sort(key=lambda item: _natural_key(item['episode_id']))
    return episodes


def _matches_task(task, patterns):
    return any(fnmatch.fnmatch(task, pattern) for pattern in patterns)


def select_episodes(episodes, patterns, max_episodes):
    tasks = sorted(episodes)
    if patterns:
        tasks = [task for task in tasks if _matches_task(task, patterns)]
        missing = [
            pattern for pattern in patterns
            if not any(_matches_task(task, [pattern]) for task in episodes)
        ]
        if missing:
            LOGGER.warning('No episodes matched --tasks entries: %s', missing)
    selected = []
    for task in tasks:
        items = episodes[task]
        if max_episodes > 0:
            items = items[:max_episodes]
        selected.extend(items)
    return selected


def probe_video(video_path):
    """Return (fps, frame_count) probed with ffprobe, then imageio."""
    ffprobe = shutil.which('ffprobe')
    if ffprobe:
        command = [
            ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=avg_frame_rate,nb_frames', '-of', 'json', video_path
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            try:
                stream = json.loads(result.stdout)['streams'][0]
            except (KeyError, IndexError, ValueError):
                stream = {}
            fps = _parse_rate(stream.get('avg_frame_rate'))
            frames = stream.get('nb_frames')
            frames = int(frames) if frames and str(frames).isdigit() else None
            if fps:
                return fps, frames
    try:
        import imageio.v3 as iio
        metadata = iio.immeta(video_path, plugin='FFMPEG')
        fps = float(metadata.get('fps') or 0)
        if fps > 0:
            return fps, None
    except Exception:  # pragma: no cover - optional decoder
        LOGGER.debug('imageio probe failed for %s', video_path, exc_info=True)
    raise RuntimeError('Cannot determine FPS for video: %s' % video_path)


def _parse_rate(value):
    if not value:
        return None
    if '/' in str(value):
        numerator, _, denominator = str(value).partition('/')
        try:
            denominator_f = float(denominator)
            if denominator_f <= 0:
                return None
            return float(numerator) / denominator_f
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def sample_indices(fps, frame_count, fps_sample, max_frames):
    """Pick frame indices at fps_sample frames/second, capped and deduped."""
    if fps <= 0 or fps_sample <= 0:
        raise ValueError('Video FPS and --fps-sample must be positive.')
    if frame_count is None or frame_count <= 0:
        raise RuntimeError('Frame count is unknown; cannot sample frames.')
    step = fps / float(fps_sample)
    count = int(math.ceil(frame_count / step))
    indices = [
        min(int(round(order * step)), frame_count - 1)
        for order in range(count)
    ]
    unique = []
    for index in indices:
        if not unique or unique[-1] != index:
            unique.append(index)
    if max_frames > 0 and len(unique) > max_frames:
        stride = len(unique) / float(max_frames)
        selected = [
            unique[min(int(round(order * stride)),
                       len(unique) - 1)] for order in range(max_frames)
        ]
        deduped = []
        for index in selected:
            if not deduped or deduped[-1] != index:
                deduped.append(index)
        unique = deduped
    if not unique:
        unique = [0]
    return unique


def extract_frames_ffmpeg(video_path,
                          indices,
                          output_dir,
                          image_max_side=448,
                          jpeg_quality=90):
    """Extract exactly the requested frames with one ffmpeg call."""
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg is not available.')
    pattern = os.path.join(output_dir, 'frame_%05d.jpg')
    for stale in glob.glob(pattern):
        os.remove(stale)
    select = '+'.join('eq(n\\,%d)' % index for index in indices)
    filters = "select='%s'" % select
    if image_max_side > 0:
        filters += (",scale=w='min(iw,%d)':h='min(ih,%d)':"
                    'force_original_aspect_ratio=decrease' %
                    (image_max_side, image_max_side))
    command = [
        ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', video_path, '-vf',
        filters, '-vsync', '0', '-q:v',
        str(jpeg_quality), pattern
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError('ffmpeg failed for %s: %s' %
                           (video_path, result.stderr.strip()))
    frames = sorted(glob.glob(pattern))
    if len(frames) != len(indices):
        raise RuntimeError('ffmpeg returned %d/%d frames for %s' %
                           (len(frames), len(indices), video_path))
    return frames


def extract_frames_imageio(video_path,
                           indices,
                           output_dir,
                           image_max_side=448,
                           jpeg_quality=90):
    """Fallback extractor used when ffmpeg is unavailable or fails."""
    import imageio.v3 as iio
    import numpy as np
    from PIL import Image
    wanted = set(int(index) for index in indices)
    captured = {}
    for order, frame in enumerate(iio.imiter(video_path, plugin='FFMPEG')):
        if order in wanted:
            captured[order] = np.asarray(frame)
        if order >= max(wanted):
            break
    missing = [index for index in indices if index not in captured]
    if missing:
        raise RuntimeError('imageio missed frames %s for %s' %
                           (missing, video_path))
    paths = []
    for order, index in enumerate(indices):
        path = os.path.join(output_dir, 'frame_%05d.jpg' % (order + 1))
        image = Image.fromarray(captured[index])
        if image_max_side > 0:
            image.thumbnail((image_max_side, image_max_side),
                            Image.Resampling.LANCZOS)
        image.convert('RGB').save(path, format='JPEG', quality=jpeg_quality)
        paths.append(path)
    return paths


def extract_frames(video_path, indices, output_dir, image_max_side,
                   jpeg_quality):
    os.makedirs(output_dir, exist_ok=True)
    try:
        return extract_frames_ffmpeg(video_path, indices, output_dir,
                                     image_max_side, jpeg_quality)
    except Exception as error:  # noqa: BLE001 - fall back to imageio
        LOGGER.debug('ffmpeg extraction failed (%s); using imageio', error)
        return extract_frames_imageio(video_path, indices, output_dir,
                                      image_max_side, jpeg_quality)


def read_hdf5_meta(hdf5_path, confidence_threshold=_ACTIVE_CONFIDENCE):
    """Read the optional EgoDex fields used for caption detail."""
    meta = dict(
        num_frames=None,
        camera_intrinsic_shape=None,
        active_hands=None,
        hand_source=None)
    try:
        import h5py
        import numpy as np
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.debug('h5py/numpy unavailable; skipping %s metadata', hdf5_path)
        return meta
    try:
        with h5py.File(hdf5_path, 'r') as handle:
            transforms = handle.get(_CAMERA_TRANSFORMS)
            if isinstance(transforms, h5py.Dataset) and transforms.ndim >= 1:
                meta['num_frames'] = int(transforms.shape[0])
            intrinsic = handle.get(_CAMERA_INTRINSIC)
            if isinstance(intrinsic, h5py.Dataset):
                meta['camera_intrinsic_shape'] = [
                    int(value) for value in intrinsic.shape
                ]
            active = {}
            for hand in _HANDS:
                group = handle.get(hand)
                if group is None:
                    continue
                active[hand] = _hand_is_active(group, np, confidence_threshold)
            known = {
                key: value
                for key, value in active.items() if value is not None
            }
            if known:
                meta['hand_source'] = 'confidences/transforms'
                left = known.get('leftHand', False)
                right = known.get('rightHand', False)
                if left and right:
                    meta['active_hands'] = 'both'
                elif left:
                    meta['active_hands'] = 'left'
                elif right:
                    meta['active_hands'] = 'right'
                else:
                    meta['active_hands'] = 'none'
    except Exception:  # noqa: BLE001 - metadata is optional
        LOGGER.debug('Cannot read %s metadata', hdf5_path, exc_info=True)
    return meta


def _hand_is_active(group, numpy, confidence_threshold):
    """Infer hand activity from confidence or wrist motion."""
    try:
        confidence = group.get('confidences')
        if confidence is not None and len(confidence):
            values = numpy.asarray(confidence[:], dtype='float32')
            return bool(float(numpy.mean(values)) >= confidence_threshold)
        transforms = group.get('transforms')
        if transforms is not None and len(transforms) > 1:
            poses = numpy.asarray(transforms[:], dtype='float32')
            # EgoDex stores a 4x4 MANO wrist transform per frame; the
            # translation column is the wrist position.
            wrist = poses[:, :3, 3]
            speed = numpy.linalg.norm(numpy.diff(wrist, axis=0), axis=-1)
            return bool(float(numpy.median(speed)) > 1e-4)
    except Exception:  # noqa: BLE001 - optional metadata
        LOGGER.debug('Cannot infer hand activity', exc_info=True)
    return None


def _pick(options, key):
    return options[zlib.crc32(key.encode('utf-8')) % len(options)]


def build_template_captions(task, meta, key):
    """Deterministic two-turn captions from task name and hand activity."""
    task_text = humanize_task(task)
    hand_phrase = _HAND_PHRASES[meta.get('active_hands')]
    question_one = _pick(_QUESTIONS_ONE, key + '/q1')
    question_two = _pick(_QUESTIONS_TWO, key + '/q2')
    answer_one = 'The person is performing the task "%s". %s' % (task_text,
                                                                 hand_phrase)
    answer_two = ('This is a first-person view of the task "%s"; %s The '
                  'sampled frame shows the current state of the '
                  'manipulation.' % (task_text, _lower_first(hand_phrase)))
    return [question_one, answer_one, question_two, answer_two]


def _lower_first(text):
    return text[:1].lower() + text[1:]


def load_vlm_captioner(module_spec, model=None):
    """Load a user captioner for --captioner vlm.

    The hook is intentionally thin: provide `module:callable` (or
    `module:Class`) where the callable has the signature
    `captioner(image_paths, meta, task) -> [q1, a1, q2, a2]`. A VLM captioner
    needs its own model and GPU budget, so it is not bundled here.
    """
    if not module_spec:
        raise SystemExit('--captioner vlm needs --captioner-module '
                         'module:callable (signature: captioner('
                         'image_paths, meta, task) -> [q1, a1, q2, a2]).')
    module_name, _, attribute = module_spec.partition(':')
    if not module_name or not attribute:
        raise SystemExit('--captioner-module must be module:callable.')
    import importlib
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if factory is None:
        raise SystemExit('Cannot find %s in %s' % (attribute, module_name))
    if isinstance(factory, type):
        return factory(model=model)
    return factory


def inspect_hdf5(path):
    """Print the HDF5 group/dataset schema of one EgoDex episode."""
    import h5py
    with h5py.File(path, 'r') as handle:
        print('HDF5:', path)
        for name in sorted(handle.keys()):
            node = handle[name]
            kind = 'dataset' if isinstance(node, h5py.Dataset) else 'group'
            print('  %s %s' % (kind, name))
            if kind == 'group':
                for child in sorted(node.keys()):
                    item = node[child]
                    if isinstance(item, h5py.Dataset):
                        print('      %s: shape=%s dtype=%s' %
                              (child, item.shape, item.dtype))
                    else:
                        print('      %s: group' % child)
            else:
                print('      shape=%s dtype=%s' % (node.shape, node.dtype))


def _episode_output_dir(images_root, task):
    return os.path.join(images_root, _safe_component(task))


def _safe_component(value):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('_') or 'unknown'


def convert_episode(episode, root, out_root, args, captioner=None):
    """Extract sampled frames and return the JSONL records for one episode."""
    meta = read_hdf5_meta(episode['hdf5'])
    fps, frame_count = probe_video(episode['video'])
    if frame_count is None:
        frame_count = meta['num_frames']
    if frame_count is None:
        raise RuntimeError('Unknown frame count for %s' % episode['video'])
    indices = sample_indices(fps, frame_count, args.fps_sample,
                             args.max_frames_per_episode)
    target = _episode_output_dir(
        os.path.join(out_root, 'images'), episode['task'])
    os.makedirs(target, exist_ok=True)
    temporary = os.path.join(target,
                             '.tmp_' + _safe_component(episode['episode_id']))
    if os.path.isdir(temporary):
        shutil.rmtree(temporary)
    os.makedirs(temporary)
    moved = []
    try:
        frames = extract_frames(episode['video'], indices, temporary,
                                args.image_max_side, args.jpeg_quality)
        records = []
        for frame_index, frame in zip(indices, frames):
            image_name = '%s_f%06d.jpg' % (episode['episode_id'].replace(
                '/', '_'), frame_index)
            final_path = os.path.join(target, image_name)
            shutil.move(frame, final_path)
            moved.append(final_path)
            image_relative = os.path.relpath(final_path,
                                             out_root).replace(os.sep, '/')
            key = '%s/%d' % (episode['episode_id'], frame_index)
            if captioner is not None:
                captions = captioner([final_path],
                                     dict(meta, task=episode['task']),
                                     episode['task'])
                if (not isinstance(captions, (list, tuple))
                        or len(captions) != 4):
                    raise ValueError(
                        'VLM captioner must return [q1, a1, q2, a2]')
                captions = [str(value) for value in captions]
            else:
                captions = build_template_captions(episode['task'], meta, key)
            records.append(
                dict(
                    id='egodex/%s/%06d' % (episode['episode_id'], frame_index),
                    image=image_relative,
                    conversations=[
                        dict(
                            **{'from': 'human'},
                            value='<image>\n' + captions[0]),
                        dict(**{'from': 'gpt'}, value=captions[1]),
                        dict(**{'from': 'human'}, value=captions[2]),
                        dict(**{'from': 'gpt'}, value=captions[3]),
                    ],
                    dataset='egodex',
                    task=episode['task'],
                    split=episode['split'],
                    episode=episode['episode_id'],
                    frame_index=int(frame_index),
                    timestamp_s=round(float(frame_index) / fps, 3),
                    source_video=os.path.relpath(episode['video'],
                                                 root).replace(os.sep, '/'),
                    source_hdf5=os.path.relpath(episode['hdf5'],
                                                root).replace(os.sep, '/'),
                    active_hands=meta.get('active_hands'),
                ))
        return records
    except Exception:
        for path in moved:
            if os.path.exists(path):
                os.remove(path)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _write_summary(out_root, summary):
    path = os.path.join(out_root, 'convert_summary.json')
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, default=str)
        stream.write('\n')
    os.replace(temporary, path)
    return path


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S')

    if args.inspect:
        inspect_hdf5(args.inspect)
        return 0
    if not args.root or not args.out:
        raise SystemExit('--root and --out are required (or use --inspect).')
    if not os.path.isdir(args.root):
        raise SystemExit('EgoDex root does not exist: %s' % args.root)
    if args.fps_sample <= 0:
        raise SystemExit('--fps-sample must be positive.')
    if args.max_frames_per_episode < 1:
        raise SystemExit('--max-frames-per-episode must be >= 1.')
    if args.workers < 1:
        raise SystemExit('--workers must be >= 1.')

    out_root = os.path.abspath(os.path.expanduser(args.out))
    jsonl_path = os.path.join(out_root, 'ego.jsonl')
    images_root = os.path.join(out_root, 'images')
    if os.path.exists(jsonl_path) or os.path.isdir(images_root):
        if not args.overwrite:
            raise SystemExit('Output already exists in %s; pass --overwrite '
                             'to replace it.' % out_root)
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
        if os.path.isdir(images_root):
            shutil.rmtree(images_root)
    os.makedirs(out_root, exist_ok=True)

    episodes = discover_episodes(
        os.path.abspath(os.path.expanduser(args.root)))
    patterns = None
    if args.tasks:
        patterns = [
            item.strip() for item in args.tasks.split(',') if item.strip()
        ]
    selected = select_episodes(episodes, patterns, args.max_episodes)
    if not selected:
        raise SystemExit('No EgoDex episodes with video were found under %s' %
                         args.root)
    tasks = sorted({episode['task'] for episode in selected})
    LOGGER.info('Converting %d episodes across %d tasks from %s',
                len(selected), len(tasks), args.root)

    captioner = None
    if args.captioner == 'vlm':
        captioner = load_vlm_captioner(args.captioner_module,
                                       args.captioner_model)
        LOGGER.info('Using VLM captioner: %s', args.captioner_module)

    temporary_jsonl = jsonl_path + '.tmp'
    per_task = {}
    written = 0
    skipped = 0
    converted = 0
    with open(temporary_jsonl, 'w', encoding='utf-8') as stream:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers) as pool:
            futures = {}
            for episode in selected:
                future = pool.submit(
                    convert_episode, episode,
                    os.path.abspath(os.path.expanduser(args.root)), out_root,
                    args, captioner)
                futures[future] = episode
            for future in concurrent.futures.as_completed(futures):
                episode = futures[future]
                converted += 1
                try:
                    records = future.result()
                except Exception as error:  # noqa: BLE001 - report and skip
                    LOGGER.warning('Skipping %s: %s', episode['episode_id'],
                                   error)
                    skipped += 1
                    continue
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stats = per_task.setdefault(episode['task'],
                                            dict(episodes=0, frames=0))
                stats['episodes'] += 1
                stats['frames'] += len(records)
                written += len(records)
                if converted % 100 == 0:
                    LOGGER.info('Converted %d/%d episodes, %d frames',
                                converted, len(selected), written)
    if written == 0:
        os.remove(temporary_jsonl)
        raise SystemExit('No frames were converted; see warnings above.')
    os.replace(temporary_jsonl, jsonl_path)

    summary = dict(
        tool='tools/egodex_to_vlm_jsonl.py',
        root=os.path.abspath(os.path.expanduser(args.root)),
        out=out_root,
        jsonl=jsonl_path,
        captioner=args.captioner,
        captioner_module=args.captioner_module,
        fps_sample=args.fps_sample,
        max_frames_per_episode=args.max_frames_per_episode,
        image_max_side=args.image_max_side,
        jpeg_quality=args.jpeg_quality,
        max_episodes=args.max_episodes,
        tasks=tasks,
        episodes=len(per_task),
        skipped_episodes=skipped,
        frames=written,
        per_task=per_task)
    summary_path = _write_summary(out_root, summary)
    LOGGER.info('Wrote %d frames from %d episodes to %s', written,
                len(per_task), jsonl_path)
    LOGGER.info('Summary: %s', summary_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
