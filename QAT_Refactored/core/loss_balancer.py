from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class LossBalanceConfig:
    """Dynamic balancing policy for deploy/KD losses."""

    strategy: str = "grad_norm"  # grad_norm | dwa | ratio
    shared_param_group: str = "head"  # head | all
    ema_decay: float = 0.95
    update_interval: int = 10
    warmup_steps: int = 0
    deploy_ramp_steps: int = 1000
    min_weight: float = 0.2
    max_weight: float = 5.0
    max_step_change: float = 1.2
    adapt_power: float = 0.5
    renorm_sum: float = 2.0
    eps: float = 1e-6

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
        if self.min_weight <= 0.0:
            raise ValueError(f"min_weight must be > 0, got {self.min_weight}")
        if self.max_weight < self.min_weight:
            raise ValueError(f"max_weight must be >= min_weight, got {self.max_weight} < {self.min_weight}")
        if self.max_step_change < 1.0:
            raise ValueError(f"max_step_change must be >= 1, got {self.max_step_change}")
        if self.adapt_power <= 0.0:
            raise ValueError(f"adapt_power must be > 0, got {self.adapt_power}")
        if self.renorm_sum <= 0.0:
            raise ValueError(f"renorm_sum must be > 0, got {self.renorm_sum}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")


class LossBalancer:
    """Stateful deploy/KD dynamic weight balancer."""

    def __init__(self, cfg: LossBalanceConfig):
        cfg.validate()
        self.cfg = cfg
        self.step = 0
        self.lambda_dep = 1.0
        self.lambda_kd = 1.0

        self._ema_dep: float | None = None
        self._ema_kd: float | None = None
        self._prev_ema_dep: float | None = None
        self._prev_ema_kd: float | None = None

    def build_total(
        self,
        deploy_loss: torch.Tensor,
        kd_loss: torch.Tensor,
        shared_params: Iterable[torch.nn.Parameter],
    ) -> tuple[torch.Tensor, float, float]:
        """Return weighted total with in-place state update."""
        self.step += 1
        dep_val = self._safe_scalar(deploy_loss)
        kd_val = self._safe_scalar(kd_loss)
        self._update_ema(dep_val, kd_val)

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

        dep_w, kd_w = self.current_weights()
        total = (deploy_loss * dep_w) + (kd_loss * kd_w)
        return total, dep_w, kd_w

    def current_weights(self) -> tuple[float, float]:
        dep = self.lambda_dep
        kd = self.lambda_kd
        if self.cfg.deploy_ramp_steps > 0 and self.step <= self.cfg.deploy_ramp_steps:
            ramp = float(self.step) / float(self.cfg.deploy_ramp_steps)
            dep = self.cfg.min_weight + (dep - self.cfg.min_weight) * max(0.0, min(1.0, ramp))
            kd = self.cfg.renorm_sum - dep
            kd = min(max(kd, self.cfg.min_weight), self.cfg.max_weight)
        return dep, kd

    def _update_ema(self, dep_val: float, kd_val: float) -> None:
        if self._ema_dep is None or self._ema_kd is None:
            self._ema_dep = dep_val
            self._ema_kd = kd_val
            self._prev_ema_dep = dep_val
            self._prev_ema_kd = kd_val
            return
        decay = self.cfg.ema_decay
        self._prev_ema_dep = self._ema_dep
        self._prev_ema_kd = self._ema_kd
        self._ema_dep = (decay * self._ema_dep) + ((1.0 - decay) * dep_val)
        self._ema_kd = (decay * self._ema_kd) + ((1.0 - decay) * kd_val)

    def _update_by_grad_norm(
        self,
        deploy_loss: torch.Tensor,
        kd_loss: torch.Tensor,
        shared_params: Iterable[torch.nn.Parameter],
    ) -> None:
        g_dep = self._grad_norm(deploy_loss, shared_params)
        g_kd = self._grad_norm(kd_loss, shared_params)
        if not math.isfinite(g_dep) or not math.isfinite(g_kd):
            return
        eps = self.cfg.eps
        p = self.cfg.adapt_power
        new_dep = self.lambda_dep * math.pow((g_kd + eps) / (g_dep + eps), p)
        new_kd = self.lambda_kd * math.pow((g_dep + eps) / (g_kd + eps), p)
        self._commit_weights(new_dep, new_kd)

    def _update_by_dwa(self) -> None:
        if self._prev_ema_dep is None or self._prev_ema_kd is None or self._ema_dep is None or self._ema_kd is None:
            return
        eps = self.cfg.eps
        p = self.cfg.adapt_power
        r_dep = self._ema_dep / (self._prev_ema_dep + eps)
        r_kd = self._ema_kd / (self._prev_ema_kd + eps)
        new_dep = self.lambda_dep * math.pow((r_dep + eps) / (r_kd + eps), p)
        new_kd = self.lambda_kd * math.pow((r_kd + eps) / (r_dep + eps), p)
        self._commit_weights(new_dep, new_kd)

    def _update_by_ratio(self) -> None:
        if self._ema_dep is None or self._ema_kd is None:
            return
        eps = self.cfg.eps
        p = self.cfg.adapt_power
        new_dep = self.lambda_dep * math.pow((self._ema_kd + eps) / (self._ema_dep + eps), p)
        new_kd = self.lambda_kd * math.pow((self._ema_dep + eps) / (self._ema_kd + eps), p)
        self._commit_weights(new_dep, new_kd)

    def _commit_weights(self, new_dep: float, new_kd: float) -> None:
        dep = self._bounded_step(self.lambda_dep, new_dep)
        kd = self._bounded_step(self.lambda_kd, new_kd)
        dep = min(max(dep, self.cfg.min_weight), self.cfg.max_weight)
        kd = min(max(kd, self.cfg.min_weight), self.cfg.max_weight)
        total = dep + kd
        if total <= self.cfg.eps:
            dep = kd = self.cfg.renorm_sum * 0.5
        else:
            scale = self.cfg.renorm_sum / total
            dep *= scale
            kd *= scale
        dep = min(max(dep, self.cfg.min_weight), self.cfg.max_weight)
        kd = min(max(kd, self.cfg.min_weight), self.cfg.max_weight)
        if not (math.isfinite(dep) and math.isfinite(kd)):
            return
        self.lambda_dep = dep
        self.lambda_kd = kd

    def _bounded_step(self, old: float, new: float) -> float:
        if not math.isfinite(new):
            return old
        max_ratio = self.cfg.max_step_change
        lo = old / max_ratio
        hi = old * max_ratio
        return min(max(new, lo), hi)

    def _grad_norm(self, loss: torch.Tensor, shared_params: Iterable[torch.nn.Parameter]) -> float:
        params = [p for p in shared_params if p.requires_grad]
        if not params:
            return 0.0
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        total = torch.zeros((), device=loss.device, dtype=torch.float32)
        for g in grads:
            if g is None:
                continue
            total = total + g.detach().float().pow(2).sum()
        return float(torch.sqrt(total + self.cfg.eps).item())

    @staticmethod
    def _safe_scalar(value: torch.Tensor) -> float:
        v = float(value.detach().float().item())
        if math.isfinite(v):
            return v
        return 0.0
