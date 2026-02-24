from typing import Callable, Iterable, Tuple
import math
from typing import List
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from torch.optim import Optimizer

def _inv_nth_root(mat: torch.Tensor, n: int, eps: float) -> torch.Tensor:
    # Compute (mat + eps I)^(-1/n) using eigendecomposition.
    d = mat.shape[0]
    eye = torch.eye(d, device=mat.device, dtype=mat.dtype)
    mat = mat + eps * eye
    e_vals, e_vecs = torch.linalg.eigh(mat)
    e_vals = torch.clamp(e_vals, min=eps)
    inv_root = e_vals.pow(-1.0 / n)
    return (e_vecs * inv_root.unsqueeze(0)) @ e_vecs.T

def _apply_preconds(grad: torch.Tensor, Ps: List[torch.Tensor]) -> torch.Tensor:
    for mode, P in enumerate(Ps):
        grad_unf = grad.movedim(mode, 0).reshape(grad.shape[mode], -1)  # (d_mode, -1)
        grad_unf = P @ grad_unf
        grad = grad_unf.reshape(grad.movedim(mode, 0).shape).movedim(0, mode)
    return grad

class Shampoo(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            update_freq: int = 10,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, update_freq=update_freq)
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
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Shampoo does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 0
                    state["Gs"] = [torch.zeros((d, d), device=p.device, dtype=p.dtype) for d in p.shape]
                    state["Ps"] = [torch.eye(d, device=p.device, dtype=p.dtype) for d in p.shape]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                state["t"] += 1

                # second-moment matrices
                for mode in range(grad.ndim):
                    g = grad.movedim(mode, 0).reshape(grad.shape[mode], -1)
                    state["Gs"][mode].add_(g @ g.T)

                # update inverse roots
                if state["t"] % uf == 0:
                    k = grad.ndim
                    n_root = 2 * k
                    state["Ps"] = [_inv_nth_root(G, n=n_root, eps=eps) for G in state["Gs"]]

                # precondition gradient
                g_pre = _apply_preconds(grad, state["Ps"])
                p.add_(g_pre, alpha=-lr)
                if wd != 0.0:
                    p.mul_(1 - lr * wd)

        return loss
