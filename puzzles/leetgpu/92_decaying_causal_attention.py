r"""
LeetGPU 92: Decaying Causal Attention
=====================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement decaying causal attention. Given query matrix `Q`, key matrix `K`,
and value matrix `V`, each of shape `seq_len × d_model`, and a scalar
decay factor `gamma` ∈ (0, 1], compute the unnormalized causal attention output
where position `n` attends to all past positions `m ≤ n` with weight
`gamman−m`:

\[
\text{output}[n] = \sum_{m=0}^{n} \gamma^{n-m} \cdot \frac{Q[n] \cdot K[m]}{\sqrt{d_{\text{model}}}} \cdot V[m]
\]

Unlike standard softmax attention, there is no normalization — the weights decay geometrically from
the current position backward. This is the parallel form of the Retention mechanism (RetNet), used
as a recurrence-friendly alternative to attention in sequence models.

(diagram omitted, see challenge.html in leetgpu-challenges)

Implementation Requirements
---------------------------

- Implement the `solve` function; do not change its signature.
- Do not use external libraries beyond those provided.
- Write the result into `output`.

Example
-------

Example 1 — with `seq_len` = 2, `d_model` = 4, `gamma` = 0.5:

\[
Q = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 1 & 1 & 0 & 0 \end{bmatrix}, \quad
K = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}, \quad
V = \begin{bmatrix} 4 & 8 & 12 & 16 \\ 4 & 8 & 12 & 16 \end{bmatrix}
\]

Attention scores \(QK^\top / \sqrt{4}\):
\[
A = \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{bmatrix}
\]
Causal decay mask \(D_{nm} = 0.5^{n-m}\) for \(n \ge m\), else \(0\):
\[
D = \begin{bmatrix} 1 & 0 \\ 0.5 & 1 \end{bmatrix}
\]
Weighted attention \(A \odot D\):
\[
\begin{bmatrix} 0.5 & 0 \\ 0.25 & 0.5 \end{bmatrix}
\]
Output \((A \odot D)\,V\):
\[
\text{output} = \begin{bmatrix} 2 & 4 & 6 & 8 \\ 3 & 6 & 9 & 12 \end{bmatrix}
\]

Constraints
-----------

- 1 ≤ `seq_len` ≤ 8,192
- 1 ≤ `d_model` ≤ 256
- 0 < `gamma` ≤ 1
- All tensors are `float32` on GPU.
- Performance is measured with `seq_len` = 4,096, `d_model` = 64

Run `python3 puzzles/leetgpu/92_decaying_causal_attention.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
import math
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, OutTensor, RandnTensor, run_challenge


class Challenge(ChallengeBase):
    name = "Decaying Causal Attention"
    atol = 0.001
    rtol = 0.001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        output: torch.Tensor,
        seq_len: int,
        d_model: int,
        gamma: float,
    ):
        assert Q.shape == (seq_len, d_model)
        assert K.shape == (seq_len, d_model)
        assert V.shape == (seq_len, d_model)
        assert output.shape == (seq_len, d_model)
        assert Q.dtype == K.dtype == V.dtype == output.dtype == torch.float32

        scale = math.sqrt(d_model)
        positions = torch.arange(seq_len, device=Q.device, dtype=Q.dtype)
        # distances[n, m] = n - m; negative means m is in the future relative to n
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        # causal: zero out future positions; clamp avoids overflow in gamma**negative
        causal = (distances >= 0).to(Q.dtype)
        decay_mask = torch.pow(gamma, distances.clamp(min=0)) * causal
        attn = torch.matmul(Q, K.T) / scale
        output.copy_(torch.matmul(attn * decay_mask, V))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "seq_len": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "gamma": (ctypes.c_float, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        # Orthogonal K rows → QK^T / sqrt(4) = [[0.5, 0.5], [0.5, 0.5]].
        # With gamma=0.5 decay mask [[1, 0], [0.5, 1]], weighted attn = [[0.5, 0], [0.25, 0.5]].
        # Output row 0 = 0.5 * V[0]; row 1 = 0.25 * V[0] + 0.5 * V[1] = [3, 6, 9, 12].
        Q = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype)
        K = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype)
        V = torch.tensor(
            [[4.0, 8.0, 12.0, 16.0], [4.0, 8.0, 12.0, 16.0]], device=device, dtype=dtype
        )
        output = torch.zeros(2, 4, device=device, dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "seq_len": 2, "d_model": 4, "gamma": 0.5}

    def _make_test_case(
        self,
        seq_len: int,
        d_model: int,
        gamma: float = 0.9,
        zero_qk: bool = False,
        negative: bool = False,
    ) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        if zero_qk:
            Q = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
            K = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
            V = torch.randn(seq_len, d_model, device=device, dtype=dtype)
        elif negative:
            Q = torch.randn(seq_len, d_model, device=device, dtype=dtype).neg()
            K = torch.randn(seq_len, d_model, device=device, dtype=dtype).neg()
            V = torch.randn(seq_len, d_model, device=device, dtype=dtype).neg()
        else:
            Q = torch.randn(seq_len, d_model, device=device, dtype=dtype)
            K = torch.randn(seq_len, d_model, device=device, dtype=dtype)
            V = torch.randn(seq_len, d_model, device=device, dtype=dtype)
        output = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "seq_len": seq_len,
            "d_model": d_model,
            "gamma": gamma,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge: single token (only self-attention possible)
        tests.append(self._make_test_case(1, 4, gamma=0.9))

        # Edge: two tokens (matches example structure)
        tests.append(self._make_test_case(2, 4, gamma=0.5))

        # Edge: gamma=1.0 — no decay, equal weight to all past positions
        tests.append(self._make_test_case(4, 8, gamma=1.0))

        # Edge: small gamma — very sharp recency bias
        tests.append(self._make_test_case(4, 8, gamma=0.1))

        # Zero Q and K: all attention scores are zero → output must be all zeros
        tests.append(self._make_test_case(8, 16, gamma=0.9, zero_qk=True))

        # All-negative Q, K, V
        tests.append(self._make_test_case(16, 16, gamma=0.8, negative=True))

        # Power-of-2 sequence length
        tests.append(self._make_test_case(32, 32, gamma=0.9))

        # Power-of-2, larger
        tests.append(self._make_test_case(64, 64, gamma=0.8))

        # Non-power-of-2 sequence length
        tests.append(self._make_test_case(30, 32, gamma=0.95))

        # Non-power-of-2, larger realistic size
        tests.append(self._make_test_case(100, 64, gamma=0.9))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # Typical LLM head: seq_len=4096, head_dim=64
        seq_len, d_model = 4096, 64
        return {
            "Q": RandnTensor((seq_len, d_model)),
            "K": RandnTensor((seq_len, d_model)),
            "V": RandnTensor((seq_len, d_model)),
            "output": OutTensor((seq_len, d_model)),
            "seq_len": seq_len,
            "d_model": d_model,
            "gamma": 0.9,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_decaying_causal_attention(...):
#     ...


def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    seq_len: int,
    d_model: int,
    gamma: float,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
