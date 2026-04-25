from __future__ import annotations
import torch

# 依你實際 repo import 路徑調整
from QAT_Refactored.core.ultralytics_kd import _compute_feature_kd_loss

def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b, h, w = 2, 80, 80
    reg_max = 16
    nc = 3
    dfl_c = 4 * reg_max
    c = dfl_c + nc

    s = torch.randn(b, c, h, w, device=device)
    t = torch.randn(b, c, h, w, device=device)

    loss = _compute_feature_kd_loss(
        student_feats=[s],
        teacher_feats=[t],
        reg_max=reg_max,
        temperature=1.0,
        cls_distill="bce",
        dfl_distill="kldiv",
        fg_threshold=0.25,
        fg_topk=500,
        fg_min_pos=200,
        fg_apply_to="both",
        device=device,
    )
    assert torch.isfinite(loss).all(), f"loss is not finite: {loss}"
    print("OK:", float(loss.detach().cpu().item()))

if __name__ == "__main__":
    main()
