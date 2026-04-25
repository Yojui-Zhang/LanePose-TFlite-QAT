from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class LossBalanceConfig:
    """Dynamic balancing policy for supervised/KD losses."""

    strategy: str = "grad_norm"  # grad_norm | dwa | ratio
    shared_param_group: str = "head"  # head | all
    ema_decay: float = 0.95
    update_interval: int = 10
    warmup_steps: int = 0
    deploy_ramp_steps: int = 1000
    min_weight: float = 0.2
    max_weight: float = 5.0
    max_step_change: float = 1.2
    adapt_power: float = 0.5  # smoothing factor in (0, 1]
    renorm_sum: float = 2.0
    eps: float = 1e-6
    fixed_kd_weight: float | None = None
    min_grad_norm: float = 0.0  # skip updates when grad signal is too weak

    def validate(self) -> None:
        if self.strategy not in {"grad_norm", "dwa", "ratio"}:
            raise ValueError(f"Unsupported balance strategy: {self.strategy}")
        if self.shared_param_group not in {"head", "all"}:
            raise ValueError(f"Unsupported shared_param_group: {self.shared_param_group}")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0,1), got {self.ema_decay}")
        if self.update_interval < 1:
            raise ValueError(f"update_interval must be >= 1, got {self.update_interval}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.deploy_ramp_steps < 0:
            raise ValueError(f"deploy_ramp_steps must be >= 0, got {self.deploy_ramp_steps}")
        if self.min_weight < 0.0:
            raise ValueError(f"min_weight must be >= 0, got {self.min_weight}")
        if self.max_weight < self.min_weight:
            raise ValueError(f"max_weight must be >= min_weight, got {self.max_weight} < {self.min_weight}")
        if self.max_step_change < 1.0:
            raise ValueError(f"max_step_change must be >= 1, got {self.max_step_change}")
        if self.adapt_power <= 0.0:
            raise ValueError(f"adapt_power must be > 0, got {self.adapt_power}")
        if not (0.0 < self.adapt_power <= 1.0):
            raise ValueError(f"adapt_power must be in (0,1], got {self.adapt_power}")
        if self.renorm_sum <= 0.0:
            raise ValueError(f"renorm_sum must be > 0, got {self.renorm_sum}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.fixed_kd_weight is not None and self.fixed_kd_weight < 0.0:
            raise ValueError(f"fixed_kd_weight must be >= 0, got {self.fixed_kd_weight}")
        if self.min_grad_norm < 0.0:
            raise ValueError(f"min_grad_norm must be >= 0, got {self.min_grad_norm}")


class LossBalancer:
    """Stateful KD-alpha balancer with fixed supervised weight = 1.0."""

    def __init__(self, cfg: LossBalanceConfig):
        cfg.validate()
        self.cfg = cfg
        self.step = 0
        self.alpha_kd = 1.0

        self._ema_sup: float | None = None
        self._ema_kd: float | None = None
        self._prev_ema_sup: float | None = None
        self._prev_ema_kd: float | None = None

        self.last_supervised_loss: float = 0.0
        self.last_kd_loss: float = 0.0
        self.last_alpha_kd: float = 1.0
        self.last_grad_norm_supervised: float | None = None
        self.last_grad_norm_kd: float | None = None
        self.last_grad_norm_ratio_sup_over_kd: float | None = None

    def build_total(
        self,
        deploy_loss: torch.Tensor,
        kd_loss: torch.Tensor,
        shared_params: Iterable[torch.nn.Parameter],
        composition: str = "dynamic_kd_deploy",
    ) -> tuple[torch.Tensor, float]:
        """Return the effective optimization scalar for the requested KD composition."""
        composition_key = str(composition).strip().lower()
        if composition_key not in {"dynamic_kd_deploy", "fixed_kd_deploy", "pure_kd"}:
            raise ValueError(f"Unsupported loss composition: {composition}")

        self.step += 1
        sup_val = self._safe_scalar(deploy_loss)
        kd_val = self._safe_scalar(kd_loss)
        self.last_supervised_loss = sup_val
        self.last_kd_loss = kd_val
        self._update_ema(sup_val, kd_val)

        if composition_key == "pure_kd":
            self.last_alpha_kd = 0.0
            return kd_loss, 0.0

        fixed_alpha = self.cfg.fixed_kd_weight
        if composition_key == "fixed_kd_deploy":
            alpha = self._bounded_alpha(1.0 if fixed_alpha is None else float(fixed_alpha))
        elif fixed_alpha is None:
            should_update = (
                self.step > self.cfg.warmup_steps
                and (self.step % self.cfg.update_interval == 0)
                and deploy_loss.requires_grad
                and kd_loss.requires_grad
            )
            if should_update:
                if self.cfg.strategy == "grad_norm":
                    self._update_by_grad_norm(deploy_loss, kd_loss, shared_params)
                elif self.cfg.strategy == "dwa":
                    self._update_by_dwa()
                else:
                    self._update_by_ratio()
            alpha = self.current_alpha()
        else:
            alpha = self._bounded_alpha(float(fixed_alpha))

        self.last_alpha_kd = alpha
        total = deploy_loss + (kd_loss * alpha)
        return total, alpha

    def current_alpha(self) -> float:
        alpha = self.alpha_kd
        if self.cfg.deploy_ramp_steps > 0 and self.step <= self.cfg.deploy_ramp_steps:
            ramp = float(self.step) / float(self.cfg.deploy_ramp_steps)
            alpha = self.cfg.min_weight + (alpha - self.cfg.min_weight) * max(0.0, min(1.0, ramp))
        return self._bounded_alpha(alpha)

    def _update_ema(self, sup_val: float, kd_val: float) -> None:
        if self._ema_sup is None or self._ema_kd is None:
            self._ema_sup = sup_val
            self._ema_kd = kd_val
            self._prev_ema_sup = sup_val
            self._prev_ema_kd = kd_val
            return
        decay = self.cfg.ema_decay
        self._prev_ema_sup = self._ema_sup
        self._prev_ema_kd = self._ema_kd
        self._ema_sup = (decay * self._ema_sup) + ((1.0 - decay) * sup_val)
        self._ema_kd = (decay * self._ema_kd) + ((1.0 - decay) * kd_val)

    def _update_by_grad_norm(
        self,
        deploy_loss: torch.Tensor,
        kd_loss: torch.Tensor,
        shared_params: Iterable[torch.nn.Parameter],
    ) -> None:
        g_sup, ok_sup = self._grad_norm(deploy_loss, shared_params)
        g_kd, ok_kd = self._grad_norm(kd_loss, shared_params)
        self.last_grad_norm_supervised = g_sup
        self.last_grad_norm_kd = g_kd
        if not (ok_sup and ok_kd):
            return
        if g_sup < self.cfg.min_grad_norm or g_kd < self.cfg.min_grad_norm:
            return
        if not (math.isfinite(g_sup) and math.isfinite(g_kd)):
            return
        eps = self.cfg.eps
        target = (g_sup + eps) / (g_kd + eps)
        if math.isfinite(target):
            self.last_grad_norm_ratio_sup_over_kd = target
        new_alpha = self._move_towards_target(self.alpha_kd, target, self.cfg.adapt_power)

        self._commit_alpha(new_alpha)

    def _update_by_dwa(self) -> None:
        if self._prev_ema_sup is None or self._prev_ema_kd is None or self._ema_sup is None or self._ema_kd is None:
            return
        eps = self.cfg.eps
        r_sup = self._ema_sup / (self._prev_ema_sup + eps)
        r_kd = self._ema_kd / (self._prev_ema_kd + eps)
        target = (r_kd + eps) / (r_sup + eps)
        new_alpha = self._move_towards_target(self.alpha_kd, target, self.cfg.adapt_power)
        self._commit_alpha(new_alpha)

    def _update_by_ratio(self) -> None:
        if self._ema_sup is None or self._ema_kd is None:
            return
        eps = self.cfg.eps
        target = (self._ema_sup + eps) / (self._ema_kd + eps)
        new_alpha = self._move_towards_target(self.alpha_kd, target, self.cfg.adapt_power)
        self._commit_alpha(new_alpha)

    @staticmethod
    def _move_towards_target(old: float, target: float, power: float) -> float:
        # \"\"\"Geometric interpolation to avoid multiplicative wind-up.\"\"\"
        if not (math.isfinite(old) and math.isfinite(target)):
            return old
        if old <= 0.0 or target <= 0.0:
            return old
        p = max(0.0, min(1.0, float(power)))
        if p == 0.0:
            return old
        if p == 1.0:
            return target
        return math.exp(((1.0 - p) * math.log(old)) + (p * math.log(target)))


    def _commit_alpha(self, new_alpha: float) -> None:
        alpha = self._bounded_step(self.alpha_kd, new_alpha)
        alpha = self._bounded_alpha(alpha)
        if not math.isfinite(alpha):
            return
        self.alpha_kd = alpha

    def _bounded_step(self, old: float, new: float) -> float:
        if not math.isfinite(new):
            return old
        max_ratio = self.cfg.max_step_change
        lo = old / max_ratio
        hi = old * max_ratio
        return min(max(new, lo), hi)

    def _bounded_alpha(self, alpha: float) -> float:
        return min(max(alpha, self.cfg.min_weight), self.cfg.max_weight)

    def _grad_norm(self, loss: torch.Tensor, shared_params: Iterable[torch.nn.Parameter]) -> tuple[float, bool]:
        params = [p for p in shared_params if p.requires_grad]
        if not params:
            return 0.0, False
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        total = torch.zeros((), device=loss.device, dtype=torch.float32)
        used = False
        for g in grads:
            if g is None:
                continue
            used = True
            total = total + g.detach().float().pow(2).sum()
        if not used:
            return 0.0, False
        norm = float(torch.sqrt(total).item())
        return norm, True

    @staticmethod
    def _safe_scalar(value: torch.Tensor) -> float:
        v = float(value.detach().float().item())
        if math.isfinite(v):
            return v
        return 0.0
