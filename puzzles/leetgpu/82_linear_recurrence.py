r"""
LeetGPU 82: Linear Recurrence
=============================

Category: ["leetgpu"]
Difficulty: ["medium"]

Given two matrices `a` and `x`, each of shape `[B, L]` (batch size × sequence length),
compute the linear recurrence `h` of shape `[B, L]` defined by:
`h[b, 0] = x[b, 0]` and `h[b, t] = a[b, t] × h[b, t−1] + x[b, t]` for `t ≥ 1`.
All values are `float32`. This operation is the core computational primitive of
State Space Models (SSMs) such as Mamba, S4, and H3.

(diagram omitted, see challenge.html in leetgpu-challenges)

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The result must be stored in the output tensor `h`

Examples
--------

Example 1 — exponential decay (`a = 0.5`, single impulse):

\[
a = \begin{bmatrix} 0.5 & 0.5 & 0.5 & 0.5 \end{bmatrix}, \quad
x = \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \end{bmatrix}
\]
\[
h = \begin{bmatrix} 1.0 & 0.5 & 0.25 & 0.125 \end{bmatrix}
\]

Example 2 — prefix sum (`a = 1`, unit inputs):

\[
a = \begin{bmatrix} 1.0 & 1.0 & 1.0 & 1.0 \end{bmatrix}, \quad
x = \begin{bmatrix} 1.0 & 1.0 & 1.0 & 1.0 \end{bmatrix}
\]
\[
h = \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 \end{bmatrix}
\]

Full example with `B = 2`, `L = 4`:

\[
a = \begin{bmatrix} 0.5 & 0.5 & 0.5 & 0.5 \\ 1.0 & 1.0 & 1.0 & 1.0 \end{bmatrix}, \quad
x = \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 1.0 & 1.0 & 1.0 & 1.0 \end{bmatrix}
\]
\[
h = \begin{bmatrix} 1.0 & 0.5 & 0.25 & 0.125 \\ 1.0 & 2.0 & 3.0 & 4.0 \end{bmatrix}
\]

Constraints
-----------

- 1 ≤ `B` ≤ 256 (batch size)
- 1 ≤ `L` ≤ 65,536 (sequence length)
- All values in `a` and `x` are `float32`
- Performance is measured with `B` = 64, `L` = 16,384

Run `python3 puzzles/leetgpu/82_linear_recurrence.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, OutTensor, RandTensor, RandnTensor, run_challenge


class Challenge(ChallengeBase):
    name = "Linear Recurrence"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        h: torch.Tensor,
        B: int,
        L: int,
    ):
        assert a.shape == (B, L)
        assert x.shape == (B, L)
        assert h.shape == (B, L)
        assert a.dtype == x.dtype == h.dtype == torch.float32

        out = torch.empty_like(x)
        out[:, 0] = x[:, 0]
        for t in range(1, L):
            out[:, t] = a[:, t] * out[:, t - 1] + x[:, t]
        h.copy_(out)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "a": (ctypes.POINTER(ctypes.c_float), "in"),
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "h": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "L": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, B, L, zero_inputs=False, zero_a=False, unit_a=False):
        device = self.device
        dtype = torch.float32
        if zero_inputs:
            a = torch.zeros(B, L, device=device, dtype=dtype)
            x = torch.zeros(B, L, device=device, dtype=dtype)
        elif zero_a:
            a = torch.zeros(B, L, device=device, dtype=dtype)
            x = torch.randn(B, L, device=device, dtype=dtype)
        elif unit_a:
            a = torch.ones(B, L, device=device, dtype=dtype)
            x = torch.randn(B, L, device=device, dtype=dtype)
        else:
            a = torch.rand(B, L, device=device, dtype=dtype)
            x = torch.randn(B, L, device=device, dtype=dtype)
        h = torch.empty(B, L, device=device, dtype=dtype)
        return {"a": a, "x": x, "h": h, "B": B, "L": L}

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        a = torch.tensor(
            [[0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        x = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        h = torch.empty(2, 4, device=device, dtype=dtype)
        return {"a": a, "x": x, "h": h, "B": 2, "L": 4}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: single element
        tests.append(self._make_test_case(1, 1))

        # Edge case: two elements
        tests.append(self._make_test_case(1, 2))

        # Zero inputs
        tests.append(self._make_test_case(4, 4, zero_inputs=True))

        # a=0 everywhere: h[t] = x[t] (no recurrence)
        tests.append(self._make_test_case(4, 16, zero_a=True))

        # a=1 everywhere: h[t] = prefix sum of x
        tests.append(self._make_test_case(4, 16, unit_a=True))

        # Power-of-2 sequence length
        tests.append(self._make_test_case(8, 32))

        # Power-of-2 sequence length, larger
        tests.append(self._make_test_case(8, 256))

        # Non-power-of-2 sequence length
        tests.append(self._make_test_case(4, 30))

        # Non-power-of-2 sequence length, larger
        tests.append(self._make_test_case(8, 100))

        # Realistic size (SSM hidden state)
        tests.append(self._make_test_case(16, 1024))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # B=64 sequences, L=16384 tokens — typical long-context SSM workload
        B, L = 64, 16384
        return {
            "a": RandTensor((B, L), 0.0, 1.0),
            "x": RandnTensor((B, L)),
            "h": OutTensor((B, L)),
            "B": B,
            "L": L,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_linear_recurrence(...):
#     ...


def solve(
    a: torch.Tensor,
    x: torch.Tensor,
    h: torch.Tensor,
    B: int,
    L: int,
):
    # TODO: launch your TileLang kernel and write the result into `h` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
