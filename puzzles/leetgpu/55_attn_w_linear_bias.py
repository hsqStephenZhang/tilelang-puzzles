r"""
LeetGPU 55: Attention with Linear Biases
========================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement Attention with Linear Biases (ALiBi), following the method described in

"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
, for a given set of matrices.
Given the query matrix `Q` of size `M×d`, key matrix `K` of size `N×d`, and value matrix
`V` of size `N×d`, your program should compute the output matrix using the formula:

$$
\text{Attention}_{ALiBi}(Q, K, V) = \text{softmax}\Bigl( \frac{QK^T}{\sqrt{d}} + \alpha \cdot \Delta \Bigr)V
$$

where α is a slope controlling the linear bias and `Δ = i - j` represents the relative position between query `i` and key `j`.
The softmax function is applied row-wise. `Q`, `K`, `V`, `output`, and `α` are all of data type `float32`;
`M`, `N`, `d` are of data type `int32`.

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
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
\(\alpha = 0.5\)

Output:

`output` (2×4):
\[
\begin{bmatrix}
3.05 & 4.05 & 6.05 & 7.05 \\
3.93 & 4.93 & 5.93 & 6.93
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
`α` = 0.8

Output:

`output` (1×2):
\[
\begin{bmatrix}
3.95 & 4.95
\end{bmatrix}
\]

Constraints
-----------

- Matrix `Q` is of size `M×d` and matrices `K` and `V` are of size
`N×d`

- 1 ≤ `M`, `N` ≤ 2048
- 1 ≤ `d` ≤ 1024
- -1.0 ≤ `α` ≤ 1.0
- Performance is measured with `M` = 2,048, `N` = 2,048

Run `python3 puzzles/leetgpu/55_attn_w_linear_bias.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, OutTensor, RandTensor, run_challenge


class Challenge(ChallengeBase):
    name = "Attention with Linear Biases"
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
        alpha: float,
    ):
        assert Q.shape == (M, d)
        assert K.shape == (N, d)
        assert V.shape == (N, d)
        assert output.shape == (M, d)

        scale = d**0.5
        attn = torch.matmul(Q, K.t()) / scale

        pos_bias = alpha * (
            torch.arange(M, device=Q.device).unsqueeze(1)
            - torch.arange(N, device=K.device).unsqueeze(0)
        )
        attn = attn + pos_bias

        attn = torch.softmax(attn, dim=1)  # M , N
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
            "alpha": (ctypes.c_float, "in"),
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
        return {"Q": Q, "K": K, "V": V, "output": output, "M": 2, "N": 3, "d": 4, "alpha": 0.5}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_example 1
        tests.append(
            {
                "Q": torch.tensor([[1.0, 2.0]], device=self.device, dtype=dtype),
                "K": torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device, dtype=dtype),
                "V": torch.tensor([[3.0, 4.0], [5.0, 6.0]], device=self.device, dtype=dtype),
                "output": torch.empty(1, 2, device=self.device, dtype=dtype),
                "M": 1,
                "N": 2,
                "d": 2,
                "alpha": 0.8,
            }
        )

        # basic_example 2
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
                "alpha": 0.5,
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
                "alpha": 0.5,
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
                "alpha": 1.0,
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
                "alpha": -0.76,
            }
        )

        # different alpha
        tests.append(
            {
                "Q": torch.empty((64, 32), device=self.device, dtype=dtype).uniform_(-1, 1),
                "K": torch.empty((128, 32), device=self.device, dtype=dtype).uniform_(-1, 1),
                "V": torch.empty((128, 32), device=self.device, dtype=dtype).uniform_(-1, 1),
                "output": torch.empty(64, 32, device=self.device, dtype=dtype),
                "M": 64,
                "N": 128,
                "d": 32,
                "alpha": -0.3,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        M, N, d = 2048, 2048, 1024
        return {
            "Q": RandTensor((M, d), -0.1, 0.1),
            "K": RandTensor((N, d), -0.1, 0.1),
            "V": RandTensor((N, d), -0.1, 0.1),
            "output": OutTensor((M, d)),
            "M": M,
            "N": N,
            "d": d,
            "alpha": 0.5,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_attn_w_linear_bias(...):
#     ...


def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    d: int,
    alpha: float,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
