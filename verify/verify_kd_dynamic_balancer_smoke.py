from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import torch

from QAT_Refactored.core.loss_balancer import LossBalanceConfig, LossBalancer


def _run_strategy_smoke(strategy: str) -> tuple[float, float]:
    cfg = LossBalanceConfig(
        strategy=strategy,
        shared_param_group="all",
        ema_decay=0.9,
        update_interval=1,
        warmup_steps=0,
        deploy_ramp_steps=0,
        min_weight=0.2,
        max_weight=5.0,
        max_step_change=1.2,
        adapt_power=0.5,
        renorm_sum=2.0,
        eps=1e-6,
    )
    balancer = LossBalancer(cfg)
    param = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
    opt = torch.optim.SGD([param], lr=0.05)

    last_dep, last_kd = 1.0, 1.0
    for _ in range(6):
        opt.zero_grad(set_to_none=True)
        deploy_loss = ((param - 2.0) ** 2).sum()
        kd_loss = (((param + 1.0) ** 2).sum()) * 3.0
        total, dep_w, kd_w = balancer.build_total(deploy_loss, kd_loss, [param])
        assert torch.isfinite(total).all(), f"{strategy}: total loss has NaN/Inf"
        assert dep_w >= cfg.min_weight and dep_w <= cfg.max_weight, f"{strategy}: dep weight out of bounds"
        assert kd_w >= cfg.min_weight and kd_w <= cfg.max_weight, f"{strategy}: kd weight out of bounds"
        total.backward()
        opt.step()
        last_dep, last_kd = dep_w, kd_w

    return float(last_dep), float(last_kd)


def main() -> None:
    dep_g, kd_g = _run_strategy_smoke("grad_norm")
    dep_d, kd_d = _run_strategy_smoke("dwa")
    dep_r, kd_r = _run_strategy_smoke("ratio")
    print(
        "verify_kd_dynamic_balancer_smoke: OK",
        f"grad_norm=({dep_g:.4f},{kd_g:.4f})",
        f"dwa=({dep_d:.4f},{kd_d:.4f})",
        f"ratio=({dep_r:.4f},{kd_r:.4f})",
    )


if __name__ == "__main__":
    main()
