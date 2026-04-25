from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

from pathlib import Path

from run_paper_experiments import _build_specs, _build_train_cmd


def main() -> None:
    specs = _build_specs(
        datasets=["kitti"],
        studies=["B"],
        seeds=[0],
        epochs=1,
        batch=2,
        imgsz=640,
        workers=0,
        close_mosaic=0,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        export_fraction=0.25,
        data_root=Path("./Paper-Data/_verify_kd_modes"),
        device_acc="0",
        device_kitti="0",
        acc_data=Path("./dataset/lanepose-carkeypoint.yaml"),
        kitti_data=Path("./dataset/KITTI.yaml"),
        acc_teacher=Path("./Teacher-model/ACC/fake"),
        kitti_teacher=Path("./Teacher-model/KITTI/fake"),
        kitti_student_model=Path("./yolo11n.pt"),
        acc_student_model=Path("./yolo11n.pt"),
        kitti_mobilenetv3_model=None,
        kitti_ghostnetv2_model=None,
        kitti_shufflenetv2_model=None,
        kitti_cira_lite_model=None,
        qat_kd_weight=None,
        qat_kd_temperature=1.0,
        qat_kd_cls_distill="bce",
        qat_kd_dfl_distill="kldiv",
        qat_kd_fg_threshold=0.25,
        qat_kd_fg_topk=800,
        qat_kd_fg_min_pos=200,
        qat_kd_fg_apply_to="both",
        include_a_cira=False,
        include_a_kitti_cira_lite=False,
        include_a_kitti_mobilenetv3=False,
        include_a_kitti_ghostnetv2=False,
        include_a_kitti_shufflenetv2=False,
        include_b_deploy_only=True,
        include_b_kd_only=True,
        include_b_pure_kd=True,
        include_b_kd_deploy=True,
        b_kd_only_weight=1.0,
        qat_balance_log_interval=20,
        qat_balance_min=0.1,
        qat_balance_max=2.0,
        qat_balance_warmup_steps=2000,
        qat_balance_max_step_change=1.2,
        qat_balance_adapt_power=0.5,
        qat_balance_strategy="grad_norm",
        qat_balance_shared_group="head",
        qat_balance_deploy_ramp_steps=800,
        qat_balance_update_interval=10,
    )

    variants = {spec.variant: spec for spec in specs}
    assert set(variants) == {"deploy_only", "KdDepoly_half", "pure_kd", "kd_deploy"}
    assert variants["deploy_only"].mode == "original"
    assert variants["KdDepoly_half"].mode == "kd-deploy"
    assert variants["pure_kd"].mode == "kd-deploy"
    assert variants["kd_deploy"].mode == "kd-deploy"
    assert variants["KdDepoly_half"].qat_balance_max == 1.0
    assert variants["KdDepoly_half"].qat_balance_warmup_steps == 0
    assert variants["KdDepoly_half"].qat_balance_deploy_ramp_steps == 0
    assert variants["KdDepoly_half"].qat_balance_update_interval == 1
    assert variants["kd_deploy"].qat_balance_max == 1.25
    assert variants["kd_deploy"].qat_balance_warmup_steps == 4000
    assert variants["kd_deploy"].qat_balance_deploy_ramp_steps == 1600
    assert variants["kd_deploy"].qat_balance_update_interval == 20

    kd_half_cmd = _build_train_cmd(variants["KdDepoly_half"], batch=2, workers=0)
    pure_kd_cmd = _build_train_cmd(variants["pure_kd"], batch=2, workers=0)
    kd_deploy_cmd = _build_train_cmd(variants["kd_deploy"], batch=2, workers=0)

    assert "--qat-kd-loss-composition" in kd_half_cmd
    assert "fixed_kd_deploy" in kd_half_cmd
    assert "--qat-kd-loss-composition" in pure_kd_cmd
    assert "pure_kd" in pure_kd_cmd
    assert "--qat-kd-loss-composition" in kd_deploy_cmd
    assert "dynamic_kd_deploy" in kd_deploy_cmd
    print("verify_run_paper_study_b_variants_smoke: OK")


if __name__ == "__main__":
    main()
