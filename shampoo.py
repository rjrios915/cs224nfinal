from typing import Callable, Iterable, List
import torch
from torch.optim import Optimizer

def _inv_nth_root(mat: torch.Tensor, n: int, eps: float) -> torch.Tensor:
    """Compute (mat + eps I)^(-1/n) via eigh, robust to numerical issues."""
    d = mat.shape[0]
    mat = mat.contiguous()
    mat = 0.5 * (mat + mat.T)  # enforce symmetry

    # bail if non-finite
    if not torch.isfinite(mat).all():
        return torch.eye(d, device=mat.device, dtype=mat.dtype)

    eye = torch.eye(d, device=mat.device, dtype=mat.dtype)
    mat = mat + eps * eye

    try:
        e_vals, e_vecs = torch.linalg.eigh(mat)
    except Exception:
        # CPU float64 fallback
        mat_cpu = mat.detach().to("cpu", dtype=torch.float64).contiguous()
        e_vals, e_vecs = torch.linalg.eigh(mat_cpu)
        e_vals = e_vals.to(device=mat.device, dtype=mat.dtype)
        e_vecs = e_vecs.to(device=mat.device, dtype=mat.dtype)

    e_vals = torch.clamp(e_vals, min=eps)
    inv_root = e_vals.pow(-1.0 / n)
    P = (e_vecs * inv_root.unsqueeze(0)) @ e_vecs.T

    if not torch.isfinite(P).all():
        return torch.eye(d, device=mat.device, dtype=mat.dtype)
    return P

def _apply_preconds(grad: torch.Tensor, Ps: List[torch.Tensor]) -> torch.Tensor:
    """Apply per-mode preconditioners (matrix or diagonal vector) to a tensor gradient."""
    for mode, P in enumerate(Ps):
        d = grad.shape[mode]
        g_unf = grad.movedim(mode, 0).reshape(d, -1)  # (d, -1)

        if P.ndim == 2:
            g_unf = P @ g_unf
        else:
            # diagonal preconditioner stored as (d,)
            g_unf = P.unsqueeze(1) * g_unf

        grad = g_unf.reshape(grad.movedim(mode, 0).shape).movedim(0, mode)
    return grad

class Shampoo(Optimizer):
    """
    Practical Shampoo for large models:
    - FP32 stats/roots
    - delayed root updates
    - diagonal fallback when any dimension > dim_threshold (paper used ~1200)
    - skips 1D params (bias, LayerNorm) by default
    """
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        eps: float = 1e-4,
        weight_decay: float = 0.0,
        update_freq: int = 20,
        dim_threshold: int = 1200,
        beta2: float = 0.9,   # EMA on second-moment stats (stability)
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} (must be >= 0)")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps} (must be >= 0)")
        if update_freq < 1:
            raise ValueError("update_freq must be >= 1")
        if dim_threshold < 1:
            raise ValueError("dim_threshold must be >= 1")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError("beta2 must be in [0, 1)")

        defaults = dict(
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            update_freq=update_freq,
            dim_threshold=dim_threshold,
            beta2=beta2,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            uf = group["update_freq"]
            thresh = group["dim_threshold"]
            beta2 = group["beta2"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("Shampoo does not support sparse gradients.")

                # Skip 1D params (LayerNorm/bias): usually not worth full Shampoo
                if grad.ndim <= 1:
                    if wd != 0.0:
                        p.mul_(1 - lr * wd)
                    p.add_(grad, alpha=-lr)
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["Gs"] = []
                    state["Ps"] = []
                    for d in p.shape:
                        if d <= thresh:
                            state["Gs"].append(torch.zeros((d, d), device=p.device, dtype=torch.float32))
                            state["Ps"].append(torch.eye(d, device=p.device, dtype=torch.float32))
                        else:
                            # diagonal fallback
                            state["Gs"].append(torch.zeros((d,), device=p.device, dtype=torch.float32))
                            state["Ps"].append(torch.ones((d,), device=p.device, dtype=torch.float32))

                state["t"] += 1

                # Update per-mode second-moment stats (EMA)
                for mode in range(grad.ndim):
                    d = grad.shape[mode]
                    g = grad.movedim(mode, 0).reshape(d, -1).float()

                    if state["Gs"][mode].ndim == 2:
                        gg = g @ g.T
                        state["Gs"][mode].mul_(beta2).add_(gg, alpha=(1 - beta2))
                    else:
                        diag = (g * g).sum(dim=1)
                        state["Gs"][mode].mul_(beta2).add_(diag, alpha=(1 - beta2))

                # Recompute inverse roots every update_freq steps
                if state["t"] % uf == 0:
                    k = grad.ndim
                    n_root = 2 * k
                    new_Ps = []
                    for G in state["Gs"]:
                        if G.ndim == 2:
                            new_Ps.append(_inv_nth_root(G, n=n_root, eps=eps))
                        else:
                            # diagonal: (G + eps)^(-1/(2k))
                            new_Ps.append(torch.clamp(G, min=0.0).add(eps).pow(-1.0 / n_root))
                    state["Ps"] = new_Ps

                # Precondition gradient in fp32, then cast back
                g_pre = _apply_preconds(grad.float(), state["Ps"]).to(dtype=grad.dtype)

                # Decoupled weight decay (apply before step)
                if wd != 0.0:
                    p.mul_(1 - lr * wd)

                p.add_(g_pre, alpha=-lr)

        return loss