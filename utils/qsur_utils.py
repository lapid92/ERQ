import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import random_hadamard_matrix
from utils.build_model import MatMul, ActQuant
# from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul, QuantAct
from copy import deepcopy

class ActCollector(nn.Module):
    def __init__(self):
        super().__init__()
        # self.register_buffer("activation_buffer", torch.zeros([10,197,384]))
        self.activation_buffer = None

    def forward(self, x):
        self.activation_buffer = x
        return x


def stat_collect_model(in_model):

    model = deepcopy(in_model)
    module_dict = {}
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(m, ActQuant):
            # Act Quant Layer
            idx = idx + 1 if idx != 0 else idx
            new_m = ActCollector()
            setattr(father_module, name[idx:], new_m)

    return model


def qsur_opt_matrix(stat_collector: nn.Module, N: 50):
    """
    """
    actquant_modules = [m.activation_buffer for m in stat_collector.modules() if isinstance(m, ActCollector)]
    all_activations = torch.cat(actquant_modules, dim=0)
    all_activations = all_activations[-N:]

    X = all_activations
    X = X.reshape(-1, X.shape[-1])  # shape: [N * 197, 384]

    # Get top-N indices
    row_var = X.var(dim=1, unbiased=False)  # shape: [num_vectors]
    top_indices = torch.topk(row_var, N, largest=True).indices
    X = X[top_indices]

    # Compute covariance
    X = X - X.mean(dim=0, keepdim=True)
    cov = X.T @ X / (X.shape[0] - 1)  # shape: [384, 384]

    # Eigen decomposition
    eigvals, Q = torch.linalg.eigh(cov)  # Q: [D, D]

    # Construct Hadamard-like matrix H (±1 entries)
    D = Q.shape[0]
    H = random_hadamard_matrix(D, device=Q.device)

    # Compute T = d^{-1/2} * H * Q^T
    # d_inv_sqrt = 1.0 / np.sqrt(D)
    # T = d_inv_sqrt * H @ Q.T  # [D, D]
    T = Q @ H.T  # [D, D]
    # T = H @ Q.T
    return T
