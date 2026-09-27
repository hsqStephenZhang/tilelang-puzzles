r"""
LeetGPU 6: Softmax Attention
============================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a GPU program that computes the softmax attention operation for a given set of matrices. Given the query
matrix `Q` of size `M×d`, key matrix `K` of size `N×d`, and value matrix
`V` of size `N×d`, your program should compute the output matrix using the formula:
$$\text{Attention}(Q, K, V) = \text{softmax}\Bigl( \frac{QK^T}{\sqrt{d}} \Bigr)V,$$ where the softmax function is
applied row-wise.

Implementation Requirements
---------------------------

- Use only GPU native features (external libraries are not permitted)
- The
`solve` function signature must remain unchanged

- The final result must be stored in the output matrix
`output`

Example 1:
----------

Input:

`Q` (2×4):
\[
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0
\end{bmatrix}
\]
`K` (3×4):
\[
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0 \\
0.0 & 0.0 & 1.0 & 0.0
\end{bmatrix}
\]
`V` (3×4):
\[
\begin{bmatrix}
1.0 & 2.0 & 3.0 & 4.0 \\
5.0 & 6.0 & 7.0 & 8.0 \\
9.0 & 10.0 & 11.0 & 12.0
\end{bmatrix}
\]

Output:

`output` (2×4):
\[
\begin{bmatrix}
4.29 & 5.29 & 6.29 & 7.29 \\
5.00 & 6.00 & 7.00 & 8.00
\end{bmatrix}
\]

Example 2:
----------

Input:

`Q` (1×2):
\[
\begin{bmatrix}
1.0 & 2.0
\end{bmatrix}
\]
`K` (2×2):
\[
\begin{bmatrix}
1.0 & 0.0 \\
0.0 & 1.0
\end{bmatrix}
\]
`V` (2×2):
\[
\begin{bmatrix}
3.0 & 4.0 \\
5.0 & 6.0
\end{bmatrix}
\]

Output:

`output` (1×2):
\[
\begin{bmatrix}
4.34 & 5.34
\end{bmatrix}
\]

Constraints
-----------

- Matrix `Q` is of size `M×d` and matrices `K` and `V` are of size
`N×d`

- 1 ≤ `M`, `N` ≤ 100,000
- 1 ≤ `d` ≤ 128
- Performance is measured with `M` = 512, `N` = 256

Run `python3 puzzles/leetgpu/6_softmax_attention.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Softmax Attention"
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
        M: int,
        N: int,
        d: int,
    ):
        scale = d**0.5
        attn = torch.matmul(Q, K.t()) / scale
        attn = torch.softmax(attn, dim=1)
        torch.matmul(attn, V, out=output)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "d": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        Q = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=dtype
        )
        K = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            device=self.device,
            dtype=dtype,
        )
        V = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            device=self.device,
            dtype=dtype,
        )
        output = torch.empty(2, 4, device=self.device, dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "M": 2, "N": 3, "d": 4}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_example
        tests.append(
            {
                "Q": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=dtype
                ),
                "K": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "V": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "output": torch.empty(2, 4, device=self.device, dtype=dtype),
                "M": 2,
                "N": 3,
                "d": 4,
            }
        )

        # zero_matrices
        tests.append(
            {
                "Q": torch.zeros((3, 5), device=self.device, dtype=dtype),
                "K": torch.zeros((3, 5), device=self.device, dtype=dtype),
                "V": torch.zeros((3, 5), device=self.device, dtype=dtype),
                "output": torch.empty(3, 5, device=self.device, dtype=dtype),
                "M": 3,
                "N": 3,
                "d": 5,
            }
        )

        # mixed_values
        tests.append(
            {
                "Q": torch.tensor(
                    [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "K": torch.tensor(
                    [[2.0, -1.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "V": torch.tensor(
                    [[1.0, 0.5, -0.5], [-1.0, 2.0, 3.0], [4.0, -2.0, 1.0], [0.0, 1.0, -1.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "output": torch.empty(4, 3, device=self.device, dtype=dtype),
                "M": 4,
                "N": 4,
                "d": 3,
            }
        )

        # large_matrices
        tests.append(
            {
                "Q": torch.empty((64, 32), device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty((128, 32), device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty((128, 32), device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "output": torch.empty(64, 32, device=self.device, dtype=dtype),
                "M": 64,
                "N": 128,
                "d": 32,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M, N, d = 512, 256, 128
        Q = torch.empty((512, 128), device=self.device, dtype=dtype).uniform_(-0.1, 0.1)
        K = torch.empty((256, 128), device=self.device, dtype=dtype).uniform_(-0.1, 0.1)
        V = torch.empty((256, 128), device=self.device, dtype=dtype).uniform_(-0.1, 0.1)
        output = torch.empty(M, d, device=self.device, dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "M": M, "N": N, "d": d}


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_softmax_attention(...):
#     ...


def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    d: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
