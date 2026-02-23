from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import torch

from QAT_Refactored.core.loss_balancer import LossBalanceConfig, LossBalancer


def _run_strategy_smoke(strategy: str) -> float:
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

    last_alpha = 1.0
    for _ in range(6):
        opt.zero_grad(set_to_none=True)
        deploy_loss = ((param - 2.0) ** 2).sum()
        kd_loss = (((param + 1.0) ** 2).sum()) * 3.0
        total, alpha_kd = balancer.build_total(deploy_loss, kd_loss, [param])
        assert torch.isfinite(total).all(), f"{strategy}: total loss has NaN/Inf"
        assert alpha_kd >= cfg.min_weight and alpha_kd <= cfg.max_weight, f"{strategy}: alpha_kd out of bounds"
        total.backward()
        opt.step()
        last_alpha = alpha_kd

    return float(last_alpha)


def _run_fixed_weight_smoke() -> float:
    cfg = LossBalanceConfig(
        strategy="grad_norm",
        shared_param_group="all",
        ema_decay=0.9,
        update_interval=1,
        warmup_steps=0,
        deploy_ramp_steps=0,
        min_weight=0.0,
        max_weight=5.0,
        max_step_change=1.2,
        adapt_power=0.5,
        renorm_sum=2.0,
        eps=1e-6,
        fixed_kd_weight=0.3,
    )
    balancer = LossBalancer(cfg)
    param = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
    deploy_loss = ((param - 2.0) ** 2).sum()
    kd_loss = (((param + 1.0) ** 2).sum()) * 3.0
    total, alpha_kd = balancer.build_total(deploy_loss, kd_loss, [param])
    assert torch.isfinite(total).all(), "fixed weight: total loss has NaN/Inf"
    assert abs(alpha_kd - 0.3) < 1e-6, f"fixed weight: expected alpha 0.3, got {alpha_kd}"
    return float(alpha_kd)


def main() -> None:
    alpha_g = _run_strategy_smoke("grad_norm")
    alpha_d = _run_strategy_smoke("dwa")
    alpha_r = _run_strategy_smoke("ratio")
    alpha_fixed = _run_fixed_weight_smoke()
    print(
        "verify_kd_dynamic_balancer_smoke: OK",
        f"grad_norm_alpha={alpha_g:.4f}",
        f"dwa_alpha={alpha_d:.4f}",
        f"ratio_alpha={alpha_r:.4f}",
        f"fixed_alpha={alpha_fixed:.4f}",
    )


if __name__ == "__main__":
    main()
