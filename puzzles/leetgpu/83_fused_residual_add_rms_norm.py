r"""
LeetGPU 83: Fused Residual Add and RMS Norm
===========================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a fused kernel that adds an input tensor `x` to a
`residual` tensor and then RMS-normalizes the result, scaling each
feature by a learned per-feature `weight` vector.
Inputs are float32 tensors: `x` and `residual` have shape
`(N, C)` where `N` is the number of tokens and `C`
is the hidden dimension, and `weight` has shape `(C,)`.
This pattern appears before every sublayer in modern LLMs such as LLaMA and Mistral:
fusing the addition and normalization into a single kernel saves a full read–write
pass over the data compared to two separate operations.

(diagram omitted, see challenge.html in leetgpu-challenges)

For each row `i` (token), the kernel must compute:

\[
z_i = x_i + \text{residual}_i, \quad
\text{rms}_i = \sqrt{\frac{1}{C}\sum_{j=0}^{C-1} z_{i,j}^2 + \varepsilon}, \quad
\text{out}_{i,j} = \frac{z_{i,j}}{\text{rms}_i} \cdot \text{weight}_j
\]

Each row's RMS is computed independently, making this a per-token normalization.
The challenge is to fuse the addition and normalization into a single GPU kernel,
avoiding writing the intermediate sum `z` to global memory.

Implementation Requirements
---------------------------

- Do not change the function signature.
- Do not use external libraries beyond what is available in the starter.
- `out[i, j]` must equal `(x[i,j] + residual[i,j]) / rms_i × weight[j]`
where `rms_i` is computed over the full row `i`.

Example
-------

N = 1, C = 4
x        = [1.0,  0.0, -1.0,  2.0]
residual = [0.5,  1.5,  0.5, -0.5]
weight   = [1.0,  1.0,  1.0,  1.0]
eps      = 1e-5

z   = [1.5,  1.5, -0.5,  1.5]
rms = sqrt((1.5^2 + 1.5^2 + 0.5^2 + 1.5^2) / 4 + 1e-5)
= sqrt(1.75 + 1e-5) ≈ 1.32288

out ≈ [1.1339,  1.1339, -0.3780,  1.1339]

Constraints
-----------

- 1 ≤ `N` ≤ 65,536
- 1 ≤ `C` ≤ 65,536
- `x`, `residual`, `weight`, and `out` are float32
- Values in `x` and `residual` are in the range [−10, 10]
- `weight` values are in the range [0.5, 1.5]
- `eps` = 1e-5
- Performance is measured with `N` = 4,096, `C` = 4,096

Run `python3 puzzles/leetgpu/83_fused_residual_add_rms_norm.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Fused Residual Add and RMS Norm"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        N: int,
        C: int,
        eps: float,
    ):
        assert x.shape == (N, C)
        assert residual.shape == (N, C)
        assert weight.shape == (C,)
        assert out.shape == (N, C)
        assert x.dtype == residual.dtype == weight.dtype == out.dtype == torch.float32

        z = x + residual
        rms = torch.sqrt(torch.mean(z**2, dim=-1, keepdim=True) + eps)
        out.copy_(z / rms * weight)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "residual": (ctypes.POINTER(ctypes.c_float), "in"),
            "weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "out": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def _make_test_case(self, N, C, eps=1e-5, zero_x=False, zero_residual=False, negative=False):
        device = self.device
        dtype = torch.float32
        if zero_x:
            x = torch.zeros(N, C, device=device, dtype=dtype)
        elif negative:
            x = torch.empty(N, C, device=device, dtype=dtype).uniform_(-2.0, -0.1)
        else:
            x = torch.randn(N, C, device=device, dtype=dtype)
        if zero_residual:
            residual = torch.zeros(N, C, device=device, dtype=dtype)
        else:
            residual = torch.randn(N, C, device=device, dtype=dtype)
        weight = torch.empty(C, device=device, dtype=dtype).uniform_(0.5, 1.5)
        out = torch.empty(N, C, device=device, dtype=dtype)
        return {
            "x": x,
            "residual": residual,
            "weight": weight,
            "out": out,
            "N": N,
            "C": C,
            "eps": eps,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        x = torch.tensor([[1.0, 0.0, -1.0, 2.0]], device=device, dtype=dtype)
        residual = torch.tensor([[0.5, 1.5, 0.5, -0.5]], device=device, dtype=dtype)
        weight = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device, dtype=dtype)
        out = torch.empty(1, 4, device=device, dtype=dtype)
        return {
            "x": x,
            "residual": residual,
            "weight": weight,
            "out": out,
            "N": 1,
            "C": 4,
            "eps": 1e-5,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: single token, single feature
        tests.append(self._make_test_case(1, 1))

        # Edge case: single token, 4 features
        tests.append(self._make_test_case(1, 4))

        # Edge case: zero x (residual pass-through with normalization)
        tests.append(self._make_test_case(2, 4, zero_x=True))

        # Edge case: zero residual (equivalent to plain RMS norm of x)
        tests.append(self._make_test_case(4, 4, zero_residual=True))

        # Negative values in x
        tests.append(self._make_test_case(4, 8, negative=True))

        # Power-of-2: typical small transformer hidden size
        tests.append(self._make_test_case(16, 64))

        # Power-of-2: medium transformer hidden size
        tests.append(self._make_test_case(32, 256))

        # Non-power-of-2
        tests.append(self._make_test_case(30, 100))

        # Non-power-of-2, larger
        tests.append(self._make_test_case(100, 255))

        # Realistic: batch of tokens, LLM-style hidden size
        tests.append(self._make_test_case(128, 512))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # 4096 tokens (e.g. batch=4, seq_len=1024), C=4096 (LLaMA-7B hidden size)
        return self._make_test_case(4096, 4096)


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_fused_residual_add_rms_norm(...):
#     ...


def solve(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    # TODO: launch your TileLang kernel and write the result into `out` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
