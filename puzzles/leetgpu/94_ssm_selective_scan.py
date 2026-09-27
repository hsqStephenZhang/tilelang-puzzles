r"""
LeetGPU 94: SSM Selective Scan
==============================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement the forward pass of a State Space Model (SSM) selective scan, the core operation in
Mamba-style sequence models. Given an input sequence `u`, time-step parameters
`delta`, state-transition matrix `A`, input projection `B`,
output projection `C`, and skip-connection weights `skip`, compute the
output sequence `y` in float32.

(diagram omitted, see challenge.html in leetgpu-challenges)

Implementation Requirements
---------------------------

Implement the function `solve(u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state)`
with the signature unchanged. Do not use external libraries beyond the allowed framework.
Write the result into the pre-allocated output tensor `y`.

For each batch `b`, position `t`, and channel `d`, the computation is:

\[
\bar{A}_{b,t,d,n} = \exp(\Delta_{b,t,d} \cdot A_{d,n})
\]
\[
\bar{B}_{b,t,d,n} = \Delta_{b,t,d} \cdot B_{b,t,n}
\]
\[
h_{b,t,d,n} = \bar{A}_{b,t,d,n} \cdot h_{b,t-1,d,n} + \bar{B}_{b,t,d,n} \cdot u_{b,t,d}
\]
\[
y_{b,t,d} = \sum_{n} C_{b,t,n} \cdot h_{b,t,d,n} + \text{skip}_d \cdot u_{b,t,d}
\]

The initial hidden state \(h_{b,-1,d,n} = 0\) for all \(b, d, n\).
All channels `d` are independent: they share the same `B` and `C`
projections but have separate state-transition rows in `A`.

Example
-------

Input:
u     = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]  shape (1,4,2)
delta = [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]  shape (1,4,2)
A     = [[-0.5, -1.0], [-0.5, -1.0]]                         shape (2,2)
B     = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]]  shape (1,4,2)
C     = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]]  shape (1,4,2)
skip  = [0.0, 0.0]                                            shape (2,)
batch=1, seq_len=4, d_model=2, d_state=2

Derivation (delta=1 everywhere, so A_bar_dn = exp(A_dn)):
A_bar[d=0] = [exp(-0.5), exp(-1.0)] ≈ [0.607, 0.368]
A_bar[d=1] = [exp(-0.5), exp(-1.0)] ≈ [0.607, 0.368]

Hidden state h has shape (d_model=2, d_state=2); initial h = zeros.
t=0: h = [[1.000, 0.000], [0.000, 0.000]]  →  y[0,0] = [1.000, 0.000]
t=1: h = [[0.607, 0.000], [0.000, 1.000]]  →  y[0,1] = [0.000, 1.000]
t=2: h = [[1.368, 1.000], [1.000, 1.368]]  →  y[0,2] = [2.368, 2.368]
t=3: h = [[0.830, 0.368], [0.607, 0.503]]  →  y[0,3] = [0.599, 0.555]

Output:
y = [[[1.000, 0.000], [0.000, 1.000], [2.368, 2.368], [0.599, 0.555]]]

Constraints
-----------

- 1 ≤ `batch` ≤ 16
- 1 ≤ `seq_len` ≤ 8,192
- 1 ≤ `d_model` ≤ 2,048
- 1 ≤ `d_state` ≤ 64
- All entries of `delta` are positive
- All entries of `A` are negative (ensuring `A_bar ∈ (0, 1)`)
- All tensors are float32 on the GPU
- Performance is measured with `batch` = 4, `seq_len` = 4,096, `d_model` = 512, `d_state` = 16

Run `python3 puzzles/leetgpu/94_ssm_selective_scan.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "SSM Selective Scan"
    atol = 0.001
    rtol = 0.001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        skip: torch.Tensor,
        y: torch.Tensor,
        batch: int,
        seq_len: int,
        d_model: int,
        d_state: int,
    ):
        assert u.shape == (batch, seq_len, d_model)
        assert delta.shape == (batch, seq_len, d_model)
        assert A.shape == (d_model, d_state)
        assert B.shape == (batch, seq_len, d_state)
        assert C.shape == (batch, seq_len, d_state)
        assert skip.shape == (d_model,)
        assert y.shape == (batch, seq_len, d_model)
        assert (
            u.dtype == delta.dtype == A.dtype == B.dtype == C.dtype == skip.dtype == torch.float32
        )

        # Hidden state: (batch, d_model, d_state)
        h = torch.zeros(batch, d_model, d_state, device=u.device, dtype=u.dtype)

        for t in range(seq_len):
            delta_t = delta[:, t, :]  # (batch, d_model)
            u_t = u[:, t, :]  # (batch, d_model)

            # Discretize: A_bar = exp(delta_t * A)
            # delta_t: (batch, d_model) -> (batch, d_model, 1)
            # A: (d_model, d_state) -> (1, d_model, d_state)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # (batch, d_model, d_state)

            # B_bar = delta_t * B_t
            # B[:, t, :]: (batch, d_state) -> (batch, 1, d_state)
            B_bar = delta_t.unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # (batch, d_model, d_state)

            # State update: h = A_bar * h + B_bar * u_t
            h = A_bar * h + B_bar * u_t.unsqueeze(-1)  # (batch, d_model, d_state)

            # Output: y_t = C_t @ h + skip * u_t
            # C[:, t, :]: (batch, d_state) -> einsum with h (batch, d_model, d_state)
            C_t = C[:, t, :]  # (batch, d_state)
            y_t = torch.einsum("bn,bdn->bd", C_t, h) + skip * u_t  # (batch, d_model)
            y[:, t, :] = y_t

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "u": (ctypes.POINTER(ctypes.c_float), "in"),
            "delta": (ctypes.POINTER(ctypes.c_float), "in"),
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "C": (ctypes.POINTER(ctypes.c_float), "in"),
            "skip": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_float), "out"),
            "batch": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "d_state": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, batch, seq_len, d_model, d_state, zero_u=False, zero_delta=False):
        device = self.device
        dtype = torch.float32
        if zero_u:
            u = torch.zeros(batch, seq_len, d_model, device=device, dtype=dtype)
        else:
            u = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
        if zero_delta:
            delta = torch.zeros(batch, seq_len, d_model, device=device, dtype=dtype)
        else:
            # delta must be positive
            delta = torch.rand(batch, seq_len, d_model, device=device, dtype=dtype) + 0.01
        # A must be negative for stability (eigenvalues < 0)
        A = -torch.rand(d_model, d_state, device=device, dtype=dtype) - 0.01
        B = torch.randn(batch, seq_len, d_state, device=device, dtype=dtype)
        C = torch.randn(batch, seq_len, d_state, device=device, dtype=dtype)
        skip = torch.rand(d_model, device=device, dtype=dtype)
        y = torch.empty(batch, seq_len, d_model, device=device, dtype=dtype)
        return {
            "u": u,
            "delta": delta,
            "A": A,
            "B": B,
            "C": C,
            "skip": skip,
            "y": y,
            "batch": batch,
            "seq_len": seq_len,
            "d_model": d_model,
            "d_state": d_state,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        device = self.device
        dtype = torch.float32
        batch, seq_len, d_model, d_state = 1, 4, 2, 2
        u = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        delta = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        A = torch.tensor([[-0.5, -1.0], [-0.5, -1.0]], device=device, dtype=dtype)
        B = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]],
            device=device,
            dtype=dtype,
        )
        C = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]],
            device=device,
            dtype=dtype,
        )
        skip = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        y = torch.empty(batch, seq_len, d_model, device=device, dtype=dtype)
        return {
            "u": u,
            "delta": delta,
            "A": A,
            "B": B,
            "C": C,
            "skip": skip,
            "y": y,
            "batch": batch,
            "seq_len": seq_len,
            "d_model": d_model,
            "d_state": d_state,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: single token
        tests.append(self._make_test_case(1, 1, 1, 4))

        # Edge case: tiny dimensions
        tests.append(self._make_test_case(1, 2, 2, 2))

        # Edge case: zero input (output should be skip * 0 = 0)
        tests.append(self._make_test_case(1, 4, 4, 4, zero_u=True))

        # Edge case: zero delta (A_bar=1, B_bar=0, so state stays zero, output = skip * u)
        tests.append(self._make_test_case(2, 4, 4, 4, zero_delta=True))

        # Power-of-2 lengths
        tests.append(self._make_test_case(2, 16, 8, 4))
        tests.append(self._make_test_case(2, 64, 16, 8))

        # Non-power-of-2
        tests.append(self._make_test_case(2, 30, 12, 4))
        tests.append(self._make_test_case(3, 100, 24, 8))

        # Typical d_state=16 (common Mamba setting)
        tests.append(self._make_test_case(2, 128, 32, 16))

        # Realistic size
        tests.append(self._make_test_case(4, 256, 64, 16))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # batch=4, seq_len=4096, d_model=512, d_state=16
        # Memory: u+delta+y ~ 3 * 4*4096*512*4 = 96MB; A+B+C+skip small
        # Total << 1GB, comfortably fits 5x in 16GB T4
        return self._make_test_case(4, 4096, 512, 16)


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_ssm_selective_scan(...):
#     ...


def solve(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    skip: torch.Tensor,
    y: torch.Tensor,
    batch: int,
    seq_len: int,
    d_model: int,
    d_state: int,
):
    # TODO: launch your TileLang kernel and write the result into `y` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
