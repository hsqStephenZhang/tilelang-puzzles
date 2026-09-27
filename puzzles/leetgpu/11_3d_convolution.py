r"""
LeetGPU 11: 3D Convolution
==========================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a program that performs a 3D convolution operation. Given a 3D input volume and a 3D kernel (filter), compute the convolved
output. The convolution should use a "valid" boundary condition (no padding).

For a 3D convolution, the output at position \((i,j,k)\) is given by:

\[
output(i,j,k) = \sum_{d=0}^{K_d-1} \sum_{r=0}^{K_r-1} \sum_{c=0}^{K_c-1} input(i+d,j+r,k+c) \cdot kernel(d,r,c)
\]

The input consists of:

-
`input`: A 3D volume of 32-bit floats, as a 1D array (row-major, then depth).

-
`kernel`: A 3D kernel of 32-bit floats, as a 1D array (row-major, then depth).

-
`input_depth`,
`input_rows`,
`input_cols`: Dimensions of the input.

-
`kernel_depth`,
`kernel_rows`,
`kernel_cols`: Dimensions of the kernel.

Output:

-
`output`: A 1D array (row-major, then depth) storing the result.

Output dimensions:

-
`output_depth = input_depth - kernel_depth + 1`

-
`output_rows = input_rows - kernel_rows + 1`

-
`output_cols = input_cols - kernel_cols + 1`

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in `output`

Examples
--------

Example 1:
----------

Input volume \(V \in \mathbb{R}^{3 \times 3 \times 3}\):
\[
\begin{aligned}
V_{d=0} &= \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix} \\
V_{d=1} &= \begin{bmatrix}
10 & 11 & 12 \\
13 & 14 & 15 \\
16 & 17 & 18
\end{bmatrix} \\
V_{d=2} &= \begin{bmatrix}
19 & 20 & 21 \\
22 & 23 & 24 \\
25 & 26 & 27
\end{bmatrix}
\end{aligned}
\]

Kernel \(K \in \mathbb{R}^{2 \times 3 \times 3}\):
\[
\begin{aligned}
K_{d=0} &= \begin{bmatrix}
1 & 0 & 0 \\
1 & 1 & 1 \\
0 & 0 & 0
\end{bmatrix} \\
K_{d=1} &= \begin{bmatrix}
1 & 1 & 0 \\
1 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\end{aligned}
\]

Output \(O \in \mathbb{R}^{2 \times 1 \times 1}\):
\[
[82, 163]
\]

Example 2:
----------

Input volume \(V \in \mathbb{R}^{2 \times 2 \times 2}\):
\[
\begin{aligned}
V_{d=0} &= \begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix} \\
V_{d=1} &= \begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix}
\end{aligned}
\]

Kernel \(K \in \mathbb{R}^{2 \times 2 \times 2}\):
\[
\begin{aligned}
K_{d=0} &= \begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix} \\
K_{d=1} &= \begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
\end{aligned}
\]

Output \(O \in \mathbb{R}^{1 \times 1 \times 1}\):
\[
[36]
\]

Constraints
-----------

- 1 ≤
`input_depth`,
`input_rows`,
`input_cols` ≤ 256

- 1 ≤
`kernel_depth`,
`kernel_rows`,
`kernel_cols` ≤ 5

-
`kernel_depth` ≤
`input_depth`

-
`kernel_rows` ≤
`input_rows`

-
`kernel_cols` ≤
`input_cols`

- Performance is measured with `input_cols` = 128, `input_rows` = 128, `kernel_cols` = 5, `kernel_rows` = 5

Run `python3 puzzles/leetgpu/11_3d_convolution.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, OutTensor, RandTensor, run_challenge


class Challenge(ChallengeBase):
    name = "3D Convolution"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        input: torch.Tensor,
        kernel: torch.Tensor,
        output: torch.Tensor,
        input_depth: int,
        input_rows: int,
        input_cols: int,
        kernel_depth: int,
        kernel_rows: int,
        kernel_cols: int,
    ):
        assert input.shape == (input_depth, input_rows, input_cols)
        assert kernel.shape == (kernel_depth, kernel_rows, kernel_cols)
        assert output.shape == (
            input_depth - kernel_depth + 1,
            input_rows - kernel_rows + 1,
            input_cols - kernel_cols + 1,
        )
        assert input.dtype == kernel.dtype == output.dtype
        assert input.device == kernel.device == output.device

        input_expanded = input.unsqueeze(0).unsqueeze(0)
        kernel_expanded = kernel.unsqueeze(0).unsqueeze(0)

        result = torch.nn.functional.conv3d(
            input_expanded, kernel_expanded, bias=None, stride=1, padding=0
        )

        output.copy_(result.squeeze(0).squeeze(0))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "kernel": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "input_depth": (ctypes.c_int, "in"),
            "input_rows": (ctypes.c_int, "in"),
            "input_cols": (ctypes.c_int, "in"),
            "kernel_depth": (ctypes.c_int, "in"),
            "kernel_rows": (ctypes.c_int, "in"),
            "kernel_cols": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input_tensor = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
            ],
            dtype=dtype,
            device=self.device,
        )
        kernel_tensor = torch.tensor(
            [[[1, 0, 0], [1, 1, 1], [0, 0, 0]], [[1, 1, 0], [1, 1, 0], [0, 0, 1]]],
            dtype=dtype,
            device=self.device,
        )
        output_tensor = torch.empty((2, 1, 1), device=self.device, dtype=dtype)
        return {
            "input": input_tensor,
            "kernel": kernel_tensor,
            "output": output_tensor,
            "input_depth": 3,
            "input_rows": 3,
            "input_cols": 3,
            "kernel_depth": 2,
            "kernel_rows": 3,
            "kernel_cols": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        device = self.device
        tests = []

        # basic_example
        tests.append(
            {
                "input": torch.tensor(
                    [
                        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
                    ],
                    dtype=dtype,
                    device=device,
                ),
                "kernel": torch.tensor(
                    [[[1, 0, 0], [1, 1, 1], [0, 0, 0]], [[1, 1, 0], [1, 1, 0], [0, 0, 1]]],
                    dtype=dtype,
                    device=device,
                ),
                "output": torch.zeros((2, 1, 1), dtype=dtype, device=device),
                "input_depth": 3,
                "input_rows": 3,
                "input_cols": 3,
                "kernel_depth": 2,
                "kernel_rows": 3,
                "kernel_cols": 3,
            }
        )

        # small_dimensions
        tests.append(
            {
                "input": torch.tensor(
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype, device=device
                ),
                "kernel": torch.tensor(
                    [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=dtype, device=device
                ),
                "output": torch.zeros((1, 1, 1), dtype=dtype, device=device),
                "input_depth": 2,
                "input_rows": 2,
                "input_cols": 2,
                "kernel_depth": 2,
                "kernel_rows": 2,
                "kernel_cols": 2,
            }
        )

        # unit_kernel
        tests.append(
            {
                "input": torch.tensor(
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype, device=device
                ),
                "kernel": torch.tensor([[[2]]], dtype=dtype, device=device),
                "output": torch.zeros((2, 2, 2), dtype=dtype, device=device),
                "input_depth": 2,
                "input_rows": 2,
                "input_cols": 2,
                "kernel_depth": 1,
                "kernel_rows": 1,
                "kernel_cols": 1,
            }
        )

        # zero_kernel
        tests.append(
            {
                "input": torch.tensor(
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype, device=device
                ),
                "kernel": torch.zeros((2, 2, 2), dtype=dtype, device=device),
                "output": torch.zeros((1, 1, 1), dtype=dtype, device=device),
                "input_depth": 2,
                "input_rows": 2,
                "input_cols": 2,
                "kernel_depth": 2,
                "kernel_rows": 2,
                "kernel_cols": 2,
            }
        )

        # negative_values
        tests.append(
            {
                "input": torch.tensor(
                    [[[-1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=dtype, device=device
                ),
                "kernel": torch.tensor([[[-1, 1], [-1, 1]]], dtype=dtype, device=device),
                "output": torch.zeros((2, 1, 1), dtype=dtype, device=device),
                "input_depth": 2,
                "input_rows": 2,
                "input_cols": 2,
                "kernel_depth": 1,
                "kernel_rows": 2,
                "kernel_cols": 2,
            }
        )

        # rectangular_dimensions
        tests.append(
            {
                "input": torch.tensor(
                    [
                        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
                    ],
                    dtype=dtype,
                    device=device,
                ),
                "kernel": torch.tensor([[[1, 1, 1], [1, 1, 1]]], dtype=dtype, device=device),
                "output": torch.zeros((2, 2, 2), dtype=dtype, device=device),
                "input_depth": 2,
                "input_rows": 3,
                "input_cols": 4,
                "kernel_depth": 1,
                "kernel_rows": 2,
                "kernel_cols": 3,
            }
        )

        # power_of_two_dimensions
        tests.append(
            {
                "input": torch.empty(4, 4, 4, device=device, dtype=dtype).uniform_(-1.0, 1.0),
                "kernel": torch.empty(3, 3, 3, device=device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.zeros((2, 2, 2), dtype=dtype, device=device),
                "input_depth": 4,
                "input_rows": 4,
                "input_cols": 4,
                "kernel_depth": 3,
                "kernel_rows": 3,
                "kernel_cols": 3,
            }
        )

        # medium_size
        tests.append(
            {
                "input": torch.empty(10, 10, 10, device=device, dtype=dtype).uniform_(-10.0, 10.0),
                "kernel": torch.empty(3, 4, 5, device=device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.zeros((8, 7, 6), dtype=dtype, device=device),
                "input_depth": 10,
                "input_rows": 10,
                "input_cols": 10,
                "kernel_depth": 3,
                "kernel_rows": 4,
                "kernel_cols": 5,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        input_depth, input_rows, input_cols = 256, 128, 128
        kernel_depth, kernel_rows, kernel_cols = 5, 5, 5
        return {
            "input": RandTensor((input_depth, input_rows, input_cols), -1.0, 1.0),
            "kernel": RandTensor((kernel_depth, kernel_rows, kernel_cols), -1.0, 1.0),
            "output": OutTensor(
                (
                    input_depth - kernel_depth + 1,
                    input_rows - kernel_rows + 1,
                    input_cols - kernel_cols + 1,
                )
            ),
            "input_depth": input_depth,
            "input_rows": input_rows,
            "input_cols": input_cols,
            "kernel_depth": kernel_depth,
            "kernel_rows": kernel_rows,
            "kernel_cols": kernel_cols,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_3d_convolution(...):
#     ...


def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_depth: int,
    input_rows: int,
    input_cols: int,
    kernel_depth: int,
    kernel_rows: int,
    kernel_cols: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
