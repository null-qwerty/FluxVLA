# Copyright 2026 Limx Dynamics
"""EgoDex frame dataset in the unified EEF action space (TASK 2, stage 1).

Each record (one JSONL line written by tools/convert_egodex_to_eef.py) holds:
  image       relative path of the extracted ego frame
  state       29D absolute EEF state [L_eef(7), L_hand(6), R_eef(7), R_hand(6), waist(3)]
              (hands/waist are zero: EgoDex has no robot hand/waist labels)
  actions     H x 29 EEF-delta targets (hands/waist zero)
  action_mask 29D per-dimension validity: 1 for the 12 arm-dim EEF entries
  task        language instruction

The dataset emits the same sample keys the RoboCasa VLA pipeline produces
(proprio / action / action_masks / images / img_masks / task_description /
stats), so the standard NormalizeStatesAndActions -> PreparePromptWithState ->
ProcessPrompts -> PadStatesAndActions chain applies unchanged.
"""
import json
import logging
import os
import tempfile
from typing import Dict, List, Optional

import numpy as np

from fluxvla.datasets.utils.vlm_annotations import IndexedJSONL
from fluxvla.engines.utils import DATASETS, TRANSFORMS

logger = logging.getLogger(__name__)

EFF_AR_DIMS = list(range(0, 6)) + list(range(13, 19))


def _check_eef_record(item):
    """Record schema of the files written by tools/convert_egodex_to_eef.py."""
    if not isinstance(item, dict) or "state" not in item or "action" not in item:
        raise ValueError("Expected an EgoDex EEF record (state/action).")
    return item


@DATASETS.register_module()
class EgoEEFDataset:
    """Read EgoDex EEF records produced by tools/convert_egodex_to_eef.py."""

    def __init__(self,
                 anno_json_path: str,
                 data_root_path: Optional[str] = None,
                 action_horizon: int = 16,
                 statistics: Optional[Dict] = None,
                 transforms: Optional[List] = None,
                 task_key: str = "task",
                 image_key: str = "image",
                 state_key: str = "state",
                 action_key: str = "action",
                 action_mask_key: str = "action_mask",
                 state_dim: int = 29,
                 seed: int = 0,
                 *args,
                 **kwargs) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        self.anno_json_path = os.path.expanduser(anno_json_path)
        self.data_root = (os.path.abspath(os.path.expanduser(data_root_path))
                          if data_root_path else
                          os.path.dirname(self.anno_json_path))
        self.action_horizon = int(action_horizon)
        self.statistics = statistics
        self.task_key = task_key
        self.image_key = image_key
        self.state_key = state_key
        self.action_key = action_key
        self.action_mask_key = action_mask_key
        self.state_dim = int(state_dim)
        self.seed = int(seed)
        self.transforms = [
            build_transform(transform) for transform in (transforms or [])
        ]
        # 192k records x ~1.5 KB would cost several GB per dataloader worker
        # (4 workers x N ranks on one node). Index the file once instead and
        # read single rows with pread; the index is shared through the node
        # local cache dir, so worker startup stays constant.
        self._index = None
        cache_dir = os.environ.get("EGO_EEF_INDEX_CACHE_DIR",
                                   os.path.join(tempfile.gettempdir(),
                                                "fluxvla_ego_eef_index"))
        try:
            self._index = IndexedJSONL(self.anno_json_path,
                                       cache_dir,
                                       validate=_check_eef_record)
        except Exception as error:  # pragma: no cover - unusual cache failures
            logger.warning("falling back to an in-memory index for %s: %s",
                           self.anno_json_path, error)
            with open(self.anno_json_path, "r", encoding="utf-8") as stream:
                self.records = [
                    _check_eef_record(json.loads(line)) for line in stream
                    if line.strip()
                ]
        if len(self) == 0:
            raise ValueError("EgoEEFDataset annotation file is empty: "
                             + self.anno_json_path)
        self._check_first_image()

    def _check_first_image(self) -> None:
        """Fail at construction with an actionable message, not inside a worker.

        Records store the frame location relative to data_root_path
        ("images/<task>.tar"), so a wrong root only shows up when the first
        batch is fetched - far from the configuration mistake that caused it.
        """
        record = self._record(0)
        relative = record.get("image_pack", record.get(self.image_key))
        if relative is None:
            raise ValueError(f"{self.anno_json_path}: first record has no "
                             f"{self.image_key}/image_pack field.")
        path = (relative if os.path.isabs(relative) else
                os.path.join(self.data_root, relative))
        if not os.path.exists(path):
            raise ValueError(
                f"{self.anno_json_path}: first frame is missing: {path} "
                f"(data_root_path={self.data_root}). Point EGO_EEF_ROOT at the "
                "directory produced by tools/convert_egodex_to_eef.py, e.g. "
                "$DATA_ROOT/egodex_eef_v1.")

    def __len__(self):
        total = len(self._index) if self._index is not None else len(
            self.records)
        return total

    def _record(self, index: int) -> Dict:
        if self._index is not None:
            return self._index[index]
        return self.records[index]

    def _load_image(self, record: Dict) -> np.ndarray:
        import io

        from PIL import Image

        if "image_pack" in record:
            # One tar per episode (see tools/convert_egodex_to_eef.py); keep the
            # last archive open because consecutive samples share an episode.
            pack = record["image_pack"]
            if not os.path.isabs(pack):
                pack = os.path.join(self.data_root, pack)
            if getattr(self, "_pack_path", None) != pack:
                if getattr(self, "_pack", None) is not None:
                    self._pack.close()
                import tarfile

                self._pack = tarfile.open(pack, "r")
                self._pack_path = pack
            member = record.get("image_member")
            if member is None:
                raise KeyError("image_member is required with image_pack")
            payload = self._pack.extractfile(member).read()
            with Image.open(io.BytesIO(payload)) as image:
                return np.asarray(image.convert("RGB"))

        relative_path = record["image"]
        path = relative_path
        if not os.path.isabs(path):
            path = os.path.join(self.data_root, path)
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    def __getitem__(self, index: int) -> Dict:
        record = self._record(index)
        state = np.asarray(record[self.state_key], dtype=np.float32)
        if state.shape != (self.state_dim, ):
            raise ValueError(
                f"{self.anno_json_path}: state shape {state.shape} "
                f"does not match state_dim={self.state_dim}.")
        # One EEF delta per record; the window walks forward inside the episode.
        horizon = self.action_horizon
        actions = np.zeros((horizon, self.state_dim), dtype=np.float32)
        masks = np.zeros((horizon, self.state_dim), dtype=np.float32)
        episode_id = record.get("episode_id")
        dim_mask = np.asarray(record[self.action_mask_key], dtype=np.float32)
        valid = 0
        for offset in range(horizon):
            position = index + offset
            if position >= len(self):
                break
            neighbour = self._record(position)
            if neighbour.get("episode_id") != episode_id:
                break
            actions[offset] = np.asarray(neighbour[self.action_key],
                                         dtype=np.float32)
            masks[offset] = dim_mask
            valid = offset + 1
        # Keys consumed by the shared RoboCasa VLA transform chain:
        # NormalizeStatesAndActions reads states/actions and looks the
        # statistics up under proprio/action.
        sample = {
            "states": state,
            "actions": actions,
            "action_masks": masks,
            "images": [self._load_image(record)],
            "img_masks": np.array([True]),
            "task_description": record.get(self.task_key, ""),
        }
        # No data_root/index key: the DictCollator only accepts the keys and
        # meta_keys of the config, and nothing in the EEF transform chain needs
        # them (frames are decoded here, not by a video transform).
        if self.statistics is not None:
            sample["stats"] = self.statistics
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def build_transform(cfg):
    """Build one transform config through the FluxVLA registry."""
    from fluxvla.engines.utils.builder import build_transform_from_cfg
    return build_transform_from_cfg(cfg)
