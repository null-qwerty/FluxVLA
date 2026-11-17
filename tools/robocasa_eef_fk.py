"""CLI wrapper for the GR1 EEF kinematics.

The implementation lives in fluxvla/utils/gr1_kinematics.py so the dataset
transforms can share it. Importing the package normally requires a GPU (the
eagle2 radio module probes CUDA at import time), so on a CPU-only conversion
node we load the module file directly instead.

Usage:
    python tools/robocasa_eef_fk.py --selftest
"""
import importlib.util
from pathlib import Path

_MODULE = Path(__file__).resolve().parents[1] / 'fluxvla' / 'utils' / 'gr1_kinematics.py'


def _load():
    try:
        from fluxvla.utils import gr1_kinematics as module
        return module
    except Exception:  # noqa: BLE001 - GPU-less conversion hosts
        spec = importlib.util.spec_from_file_location('gr1_kinematics', _MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


kin = _load()
GR1Kinematics = kin.GR1Kinematics
eef_abs_vector = kin.eef_abs_vector
eef_delta_vector = kin.eef_delta_vector
apply_eef_delta = kin.apply_eef_delta
to_camera_frame = kin.to_camera_frame
joint29_from_parquet_row = kin.joint29_from_parquet_row


if __name__ == '__main__':
    raise SystemExit(kin.main())
