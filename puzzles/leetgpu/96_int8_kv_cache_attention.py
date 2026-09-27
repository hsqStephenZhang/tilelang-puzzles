r"""
LeetGPU 96: INT8 KV-Cache Attention
===================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement decode-phase multi-head attention where the key and value caches are stored as
`int8` with per-token scale factors. This memory layout halves KV-cache bandwidth
versus `float32` and is used in production LLM serving systems such as TensorRT-LLM
and vLLM. Given a query tensor `Q` for a single new token, `int8` key cache
`K_int8`, `int8` value cache `V_int8`, and per-token scales
`k_scale` and `v_scale`, dequantize the caches and compute scaled
dot-product attention to produce `output`. All non-integer tensors use
`float32`.

Implementation Requirements
---------------------------

- Implement the function `solve(Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim)`.
- Do not change the function signature or use external libraries beyond the standard GPU frameworks.
- Write the result into the provided `output` buffer.
- Dequantize using per-token scales: `K_float[h, s, d] = K_int8[h, s, d] × k_scale[h, s]` (and analogously for V).
- Use scaled dot-product attention with scale factor `1 / sqrt(head_dim)` and a softmax over the sequence dimension.

Example
-------

With `num_heads` = 1, `seq_len` = 3, `head_dim` = 4:

Input:

\(Q\) (1×4):
\[
\begin{bmatrix} 1 & 1 & 1 & 1 \end{bmatrix}
\]
\(K\_int8\) (1×3×4):
\[
\begin{bmatrix} 10 & 0 & 0 & 0 \\ 0 & 10 & 0 & 0 \\ 0 & 0 & 10 & 0 \end{bmatrix}
\]
\(k\_scale\) (1×3): \(\begin{bmatrix} 0.1 & 0.1 & 0.1 \end{bmatrix}\)
⇒
\(K\_float\) (1×3×4):
\[
\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}
\]
\(V\_int8\) (1×3×4):
\[
\begin{bmatrix} 10 & 20 & 30 & 40 \\ 50 & 60 & 70 & 80 \\ 90 & 100 & 110 & 120 \end{bmatrix}
\]
\(v\_scale\) (1×3): \(\begin{bmatrix} 0.1 & 0.1 & 0.1 \end{bmatrix}\)
⇒
\(V\_float\) (1×3×4):
\[
\begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \end{bmatrix}
\]

Scores = \(Q \cdot K\_float^T / \sqrt{4}\) = \(\begin{bmatrix} 0.5 & 0.5 & 0.5 \end{bmatrix}\),
so softmax weights = \(\begin{bmatrix} 1/3 & 1/3 & 1/3 \end{bmatrix}\).

Output (1×4):
\[
\begin{bmatrix} 5.00 & 6.00 & 7.00 & 8.00 \end{bmatrix}
\]

Constraints
-----------

- 1 ≤ `num_heads` ≤ 64
- 1 ≤ `seq_len` ≤ 32,768
- 8 ≤ `head_dim` ≤ 256; `head_dim` is a multiple of 8
- `K_int8` and `V_int8` values are in \([-128, 127]\)
- All scale values are positive `float32`
- Performance is measured with `num_heads` = 32, `seq_len` = 8,192, `head_dim` = 128

Run `python3 puzzles/leetgpu/96_int8_kv_cache_attention.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
import math
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "INT8 KV-Cache Attention"
    atol = 0.001
    rtol = 0.001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K_int8: torch.Tensor,
        V_int8: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        output: torch.Tensor,
        num_heads: int,
        seq_len: int,
        head_dim: int,
    ):
        assert Q.shape == (num_heads, head_dim)
        assert K_int8.shape == (num_heads, seq_len, head_dim)
        assert V_int8.shape == (num_heads, seq_len, head_dim)
        assert k_scale.shape == (num_heads, seq_len)
        assert v_scale.shape == (num_heads, seq_len)
        assert output.shape == (num_heads, head_dim)
        assert Q.dtype == torch.float32
        assert K_int8.dtype == torch.int8
        assert V_int8.dtype == torch.int8
        assert k_scale.dtype == torch.float32
        assert v_scale.dtype == torch.float32
        assert output.dtype == torch.float32

        # Dequantize: K_float[h, s, d] = K_int8[h, s, d] * k_scale[h, s]
        K_float = K_int8.float() * k_scale.unsqueeze(-1)  # [num_heads, seq_len, head_dim]
        V_float = V_int8.float() * v_scale.unsqueeze(-1)  # [num_heads, seq_len, head_dim]

        # Scaled dot-product attention: Q [num_heads, head_dim] attends to all seq_len positions
        scale = 1.0 / math.sqrt(head_dim)
        # scores: [num_heads, 1, seq_len]
        scores = torch.bmm(Q.unsqueeze(1), K_float.transpose(1, 2)) * scale
        weights = torch.softmax(scores, dim=-1)  # [num_heads, 1, seq_len]

        # Weighted sum of V: [num_heads, 1, seq_len] @ [num_heads, seq_len, head_dim]
        out = torch.bmm(weights, V_float)  # [num_heads, 1, head_dim]
        output.copy_(out.squeeze(1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K_int8": (ctypes.POINTER(ctypes.c_int8), "in"),
            "V_int8": (ctypes.POINTER(ctypes.c_int8), "in"),
            "k_scale": (ctypes.POINTER(ctypes.c_float), "in"),
            "v_scale": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_heads": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, num_heads, seq_len, head_dim, zero_q=False, seed=None):
        device = self.device
        if seed is not None:
            torch.manual_seed(seed)
        if zero_q:
            Q = torch.zeros(num_heads, head_dim, dtype=torch.float32, device=device)
        else:
            Q = torch.randn(num_heads, head_dim, dtype=torch.float32, device=device)
        K_int8 = torch.randint(
            -128, 128, (num_heads, seq_len, head_dim), dtype=torch.int8, device=device
        )
        V_int8 = torch.randint(
            -128, 128, (num_heads, seq_len, head_dim), dtype=torch.int8, device=device
        )
        k_scale = torch.rand(num_heads, seq_len, dtype=torch.float32, device=device) * 0.1 + 0.01
        v_scale = torch.rand(num_heads, seq_len, dtype=torch.float32, device=device) * 0.1 + 0.01
        output = torch.empty(num_heads, head_dim, dtype=torch.float32, device=device)
        return {
            "Q": Q,
            "K_int8": K_int8,
            "V_int8": V_int8,
            "k_scale": k_scale,
            "v_scale": v_scale,
            "output": output,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        num_heads, seq_len, head_dim = 1, 3, 4
        Q = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
        K_int8 = torch.tensor(
            [[[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0]]], dtype=torch.int8, device=device
        )
        V_int8 = torch.tensor(
            [[[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]],
            dtype=torch.int8,
            device=device,
        )
        k_scale = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32, device=device)
        v_scale = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32, device=device)
        output = torch.empty(num_heads, head_dim, dtype=torch.float32, device=device)
        return {
            "Q": Q,
            "K_int8": K_int8,
            "V_int8": V_int8,
            "k_scale": k_scale,
            "v_scale": v_scale,
            "output": output,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []
        # Edge: single key in cache
        tests.append(self._make_test_case(1, 1, 8, seed=0))
        # Edge: two keys
        tests.append(self._make_test_case(1, 2, 8, seed=1))
        # Edge: four keys, two heads
        tests.append(self._make_test_case(2, 4, 8, seed=2))
        # Zero query (uniform softmax weights)
        tests.append(self._make_test_case(1, 8, 16, zero_q=True, seed=3))
        # Power-of-2 seq_len
        tests.append(self._make_test_case(4, 16, 64, seed=4))
        tests.append(self._make_test_case(8, 64, 64, seed=5))
        # Non-power-of-2
        tests.append(self._make_test_case(2, 30, 64, seed=6))
        tests.append(self._make_test_case(4, 100, 64, seed=7))
        # Realistic sizes
        tests.append(self._make_test_case(16, 512, 64, seed=8))
        tests.append(self._make_test_case(32, 256, 128, seed=9))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make_test_case(32, 8192, 128, seed=42)


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_int8_kv_cache_attention(...):
#     ...


def solve(
    Q: torch.Tensor,
    K_int8: torch.Tensor,
    V_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    seq_len: int,
    head_dim: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
