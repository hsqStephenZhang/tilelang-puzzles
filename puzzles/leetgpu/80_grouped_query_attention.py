r"""
LeetGPU 80: Grouped Query Attention
===================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement Grouped Query Attention (GQA), the attention mechanism used in modern large language
models such as LLaMA-3, Mistral, and Gemma. GQA reduces the KV-cache memory footprint during
inference by sharing key and value heads across groups of query heads. Given query tensor
`Q` with `num_q_heads` heads and key/value tensors `K`,
`V` each with `num_kv_heads` heads, compute scaled dot-product attention
where every group of `num_q_heads / num_kv_heads` consecutive query heads attends to
the same key and value head. All tensors use `float32`.

(diagram omitted, see challenge.html in leetgpu-challenges)

Implementation Requirements
---------------------------

- Implement the function `solve(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim)`.
- Do not change the function signature or use external libraries beyond the standard GPU frameworks.
- Write the result into the provided `output` buffer.
- `num_q_heads` is always divisible by `num_kv_heads`.
- Use scaled dot-product attention with scale factor `1 / sqrt(head_dim)` and a softmax over the key dimension.

Example
-------

With `num_q_heads` = 4, `num_kv_heads` = 2 (groups of 2), `seq_len` = 3,
`head_dim` = 4:

Input:

\(Q_0\) (3×4):
\[
\begin{bmatrix}
1 & 0 & 0 & 1 \\
0 & 1 & 1 & 0 \\
1 & 1 & 0 & 0
\end{bmatrix}
\]
\(Q_1\) (3×4):
\[
\begin{bmatrix}
0 & 1 & 0 & 1 \\
1 & 0 & 1 & 0 \\
0 & 0 & 1 & 1
\end{bmatrix}
\]
\(Q_2\) (3×4):
\[
\begin{bmatrix}
-1 & 0 & 0.5 & 0 \\
0 & -1 & 0 & 0.5 \\
0.5 & 0 & -1 & 0
\end{bmatrix}
\]
\(Q_3\) (3×4):
\[
\begin{bmatrix}
0 & 0.5 & 0 & -1 \\
0.5 & 0 & 0 & -1 \\
0 & 0 & 0.5 & 0.5
\end{bmatrix}
\]
\(K_0\) (3×4):
\[
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
1 & 1 & 1 & 1
\end{bmatrix}
\]
\(K_1\) (3×4):
\[
\begin{bmatrix}
0 & 1 & 0 & -1 \\
-1 & 0 & 1 & 0 \\
0 & -1 & 0 & 1
\end{bmatrix}
\]
\(V_0\) (3×4):
\[
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12
\end{bmatrix}
\]
\(V_1\) (3×4):
\[
\begin{bmatrix}
-1 & -2 & -3 & -4 \\
2 & 3 & 4 & 5 \\
6 & 7 & 8 & 9
\end{bmatrix}
\]
Groups: \(Q_0, Q_1 \to K_0, V_0\); \quad \(Q_2, Q_3 \to K_1, V_1\)

Output (values rounded to 2 decimal places):

\(\text{output}_0\) (3×4):
\[
\begin{bmatrix}
5.71 & 6.71 & 7.71 & 8.71 \\
5.71 & 6.71 & 7.71 & 8.71 \\
5.71 & 6.71 & 7.71 & 8.71
\end{bmatrix}
\]
\(\text{output}_1\) (3×4):
\[
\begin{bmatrix}
6.07 & 7.07 & 8.07 & 9.07 \\
5.00 & 6.00 & 7.00 & 8.00 \\
5.71 & 6.71 & 7.71 & 8.71
\end{bmatrix}
\]
\(\text{output}_2\) (3×4):
\[
\begin{bmatrix}
2.24 & 2.76 & 3.27 & 3.79 \\
3.96 & 4.70 & 5.44 & 6.17 \\
2.40 & 2.60 & 2.79 & 2.98
\end{bmatrix}
\]
\(\text{output}_3\) (3×4):
\[
\begin{bmatrix}
0.76 & 0.58 & 0.40 & 0.22 \\
1.17 & 1.08 & 1.00 & 0.91 \\
2.84 & 3.37 & 3.91 & 4.44
\end{bmatrix}
\]

Constraints
-----------

- 1 ≤ `num_kv_heads` ≤ `num_q_heads` ≤ 64
- `num_q_heads` is divisible by `num_kv_heads`
- 1 ≤ `seq_len` ≤ 4,096
- 8 ≤ `head_dim` ≤ 256; `head_dim` is a multiple of 8
- All tensor values are `float32`
- Performance is measured with `num_q_heads` = 32, `num_kv_heads` = 8, `seq_len` = 1,024, `head_dim` = 128

Run `python3 puzzles/leetgpu/80_grouped_query_attention.py` to test your `solve`, or add `--ref` to run the
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
    name = "Grouped Query Attention"
    atol = 0.0001
    rtol = 0.0001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        output: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
    ):
        assert Q.shape == (num_q_heads, seq_len, head_dim)
        assert K.shape == (num_kv_heads, seq_len, head_dim)
        assert V.shape == (num_kv_heads, seq_len, head_dim)
        assert output.shape == (num_q_heads, seq_len, head_dim)
        assert Q.dtype == K.dtype == V.dtype == output.dtype == torch.float32
        assert num_q_heads % num_kv_heads == 0

        num_groups = num_q_heads // num_kv_heads
        scale = 1.0 / math.sqrt(head_dim)

        # Expand K, V from (num_kv_heads, seq_len, head_dim)
        # to (num_q_heads, seq_len, head_dim) by repeating each KV head num_groups times
        K_expanded = K.repeat_interleave(num_groups, dim=0)
        V_expanded = V.repeat_interleave(num_groups, dim=0)

        # Scaled dot-product attention: (num_q_heads, seq_len, seq_len)
        scores = torch.bmm(Q, K_expanded.transpose(1, 2)) * scale

        # Softmax over the key dimension
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values: (num_q_heads, seq_len, head_dim)
        output.copy_(torch.bmm(attn_weights, V_expanded))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_q_heads": (ctypes.c_int, "in"),
            "num_kv_heads": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, num_q_heads, num_kv_heads, seq_len, head_dim, zero_inputs=False):
        dtype = torch.float32
        device = self.device
        if zero_inputs:
            Q = torch.zeros(num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
            K = torch.zeros(num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
            V = torch.zeros(num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        else:
            Q = torch.randn(num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
            K = torch.randn(num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
            V = torch.randn(num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        output = torch.zeros(num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        dtype = torch.float32
        device = self.device
        num_q_heads = 4
        num_kv_heads = 2
        seq_len = 3
        head_dim = 4

        Q = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
                [[-1.0, 0.0, 0.5, 0.0], [0.0, -1.0, 0.0, 0.5], [0.5, 0.0, -1.0, 0.0]],
                [[0.0, 0.5, 0.0, -1.0], [0.5, 0.0, 0.0, -1.0], [0.0, 0.0, 0.5, 0.5]],
            ],
            device=device,
            dtype=dtype,
        )
        K = torch.tensor(
            [
                [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]],
            ],
            device=device,
            dtype=dtype,
        )
        V = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                [[-1.0, -2.0, -3.0, -4.0], [2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]],
            ],
            device=device,
            dtype=dtype,
        )
        output = torch.zeros(num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: MQA (num_kv_heads=1), single token
        tests.append(self._make_test_case(4, 1, 1, 8))

        # Edge case: GQA with groups=2, tiny seq
        tests.append(self._make_test_case(2, 1, 2, 4))

        # Zero inputs
        tests.append(self._make_test_case(4, 2, 4, 8, zero_inputs=True))

        # Power-of-2: groups=4 (LLaMA-3 style ratio)
        tests.append(self._make_test_case(8, 2, 16, 32))

        # Power-of-2: seq_len=32, head_dim=64
        tests.append(self._make_test_case(4, 2, 32, 64))

        # Non-power-of-2 seq_len
        tests.append(self._make_test_case(4, 2, 30, 32))

        # Non-power-of-2 seq_len, different grouping
        tests.append(self._make_test_case(6, 3, 100, 32))

        # GQA groups=8 (Mistral style), seq_len=255
        tests.append(self._make_test_case(8, 1, 255, 64))

        # MHA equivalent (num_q_heads == num_kv_heads)
        tests.append(self._make_test_case(8, 8, 64, 32))

        # Realistic small inference batch
        tests.append(self._make_test_case(8, 2, 128, 64))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # LLaMA-3 8B style: 32 Q heads, 8 KV heads, head_dim=128
        num_q_heads, num_kv_heads, seq_len, head_dim = 32, 8, 1024, 128
        return {
            "Q": RandnTensor((num_q_heads, seq_len, head_dim)),
            "K": RandnTensor((num_kv_heads, seq_len, head_dim)),
            "V": RandnTensor((num_kv_heads, seq_len, head_dim)),
            "output": OutTensor((num_q_heads, seq_len, head_dim)),
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_grouped_query_attention(...):
#     ...


def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
