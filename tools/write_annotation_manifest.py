#!/usr/bin/env python3
"""Write the sha256/size manifest that `scripts/_preflight_index.py` verifies.

The JSONL byte-offset index is keyed by a fingerprint of the file metadata.
On a shared CPFS mount a node can briefly see a stale copy of a recently
written annotation file, which either aborts the index build or silently
indexes a truncated view. The manifest records the authoritative sha256/size
so each node can verify it sees the same bytes before training.

Run it on the machine that wrote the JSONL (i.e. the data-processing host),
never from a training node:

    python tools/write_annotation_manifest.py /path/ego_train.jsonl /path/ego_val.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os


def manifest_for(path):
    digest = hashlib.sha256()
    rows = 0
    with open(path, 'rb') as stream:
        for line in stream:
            digest.update(line)
            if line.strip():
                rows += 1
    return dict(sha256=digest.hexdigest(),
                size=os.path.getsize(path),
                rows=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jsonl', nargs='+', help='annotation JSONL file(s)')
    args = parser.parse_args(argv)
    for path in args.jsonl:
        data = manifest_for(path)
        out = path + '.manifest.json'
        with open(out, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2)
            stream.write('\n')
        print(f'{out}: sha256={data["sha256"][:16]}... size={data["size"]} rows={data["rows"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
