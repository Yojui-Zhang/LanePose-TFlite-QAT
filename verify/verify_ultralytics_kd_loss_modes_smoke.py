from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import torch

from QAT_Refactored.core.loss_balancer import LossBalanceConfig, LossBalancer


def main() -> None:
    shared = [torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))]
    deploy_loss = torch.tensor(2.0, requires_grad=True)
    kd_loss = torch.tensor(3.0, requires_grad=True)

    fixed_balancer = LossBalancer(LossBalanceConfig(fixed_kd_weight=1.5))
    fixed_total, fixed_alpha = fixed_balancer.build_total(
        deploy_loss=deploy_loss,
        kd_loss=kd_loss,
        shared_params=shared,
        composition="fixed_kd_deploy",
    )
    assert torch.allclose(fixed_total, torch.tensor(6.5))
    assert fixed_alpha == 1.5

    pure_balancer = LossBalancer(LossBalanceConfig(fixed_kd_weight=1.5))
    pure_total, pure_alpha = pure_balancer.build_total(
        deploy_loss=deploy_loss,
        kd_loss=kd_loss,
        shared_params=shared,
        composition="pure_kd",
    )
    assert torch.allclose(pure_total, torch.tensor(3.0))
    assert pure_alpha == 0.0
    assert pure_balancer.last_supervised_loss == 2.0
    assert pure_balancer.last_kd_loss == 3.0
    print("verify_ultralytics_kd_loss_modes_smoke: OK")


if __name__ == "__main__":
    main()
