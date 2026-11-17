"""CLI wrapper for the GR1 EEF kinematics (implemented in the package so the
dataset transforms can share it).

Usage:
    python tools/robocasa_eef_fk.py --selftest
"""
from fluxvla.utils.gr1_kinematics import (  # noqa: F401
    GR1Kinematics, apply_eef_delta, eef_abs_vector, eef_delta_vector,
    joint29_from_parquet_row, to_camera_frame)


if __name__ == '__main__':
    from fluxvla.utils.gr1_kinematics import main
    raise SystemExit(main())
