from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

from pathlib import Path

from train_pose import _build_arg_parser, _build_kd_deploy_overrides, _parse_imgsz


def main() -> None:
    parser = _build_arg_parser()
    teacher_dir = Path("./verify/_tmp/fake_teacher")
    teacher_dir.mkdir(parents=True, exist_ok=True)

    fixed_args = parser.parse_args(
        [
            "--data",
            "./dataset/KITTI.yaml",
            "--task",
            "detect",
            "--qat-loss-mode",
            "kd-deploy",
            "--qat-kd-loss-composition",
            "fixed_kd_deploy",
            "--qat-kd-weight",
            "1.0",
            "--imgsz",
            "640",
            "640",
            "--epochs",
            "1",
            "--batch",
            "2",
            "--workers",
            "0",
            "--qat-teacher-exported-dir",
            str(teacher_dir),
        ]
    )
    fixed_overrides, _ = _build_kd_deploy_overrides(fixed_args, _parse_imgsz(fixed_args.imgsz))
    assert fixed_overrides["ULTRA_KD_LOSS_COMPOSITION"] == "fixed_kd_deploy"

    pure_args = parser.parse_args(
        [
            "--data",
            "./dataset/KITTI.yaml",
            "--task",
            "detect",
            "--qat-loss-mode",
            "kd-deploy",
            "--qat-kd-loss-composition",
            "pure_kd",
            "--imgsz",
            "640",
            "640",
            "--epochs",
            "1",
            "--batch",
            "2",
            "--workers",
            "0",
            "--qat-teacher-exported-dir",
            str(teacher_dir),
        ]
    )
    pure_overrides, _ = _build_kd_deploy_overrides(pure_args, _parse_imgsz(pure_args.imgsz))
    assert pure_overrides["ULTRA_KD_LOSS_COMPOSITION"] == "pure_kd"

    pure_missing_teacher_args = parser.parse_args(
        [
            "--data",
            "./dataset/KITTI.yaml",
            "--task",
            "detect",
            "--qat-loss-mode",
            "kd-deploy",
            "--qat-kd-loss-composition",
            "pure_kd",
            "--imgsz",
            "640",
            "640",
            "--epochs",
            "1",
            "--batch",
            "2",
            "--workers",
            "0",
        ]
    )
    try:
        _build_kd_deploy_overrides(pure_missing_teacher_args, _parse_imgsz(pure_missing_teacher_args.imgsz))
    except ValueError as exc:
        assert "teacher" in str(exc).lower()
    else:
        raise AssertionError("pure_kd without teacher should fail fast")

    print("verify_train_pose_kd_loss_modes_smoke: OK")


if __name__ == "__main__":
    main()
