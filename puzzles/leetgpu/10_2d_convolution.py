r"""
LeetGPU 10: 2D Convolution
==========================

Category: ["leetgpu"]
Difficulty: ["medium"]

Write a program that performs a 2D convolution operation on the GPU. Given an input matrix and a kernel (filter), compute the convolved
output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied
where it fully overlaps with the input.

(diagram omitted, see challenge.html in leetgpu-challenges)

The input consists of:

- `input`: A 2D matrix of 32-bit floating-point numbers, represented as a 1D array in row-major order.
- `kernel`: A 2D kernel (filter) of 32-bit floating-point numbers, also represented as a 1D array in
row-major order.

The output should be written to the `output` matrix (also a 1D array in row-major order). The output matrix will have dimensions:

- `output_rows = input_rows - kernel_rows + 1`
- `output_cols = input_cols - kernel_cols + 1`

The convolution operation is defined as:

\(output[i][j] = \sum_{m=0}^{kernel\_rows-1} \sum_{n=0}^{kernel\_cols-1} input[i+m][j+n] * kernel[m][n]\)

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The
`solve` function signature must remain unchanged

- The final result must be stored in the array
`output`

Example 1:
----------

Input:

`input` (3×3):
\[
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
\]
`kernel` (2×2):
\[
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\]
`input_rows = 3`

`input_cols = 3`

`kernel_rows = 2`

`kernel_cols = 2`

Output:

`output` (2×2):
\[
\begin{bmatrix}
6 & 8 \\
12 & 14
\end{bmatrix}
\]

Example 2:
----------

Input:

`input` (4×4):
\[
\begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 2 & 3 & 1 \\
1 & 4 & 5 & 1 \\
1 & 1 & 1 & 1
\end{bmatrix}
\]
`kernel` (1×3):
\[
\begin{bmatrix}
1 & 0 & 1
\end{bmatrix}
\]
`input_rows = 4`

`input_cols = 4`

`kernel_rows = 1`

`kernel_cols = 3`

Output:

`output` (4×2):
\[
\begin{bmatrix}
2 & 2 \\
4 & 3 \\
6 & 5 \\
2 & 2
\end{bmatrix}
\]

Constraints
-----------

- 1 ≤ `input_rows`, `input_cols` ≤ 3072
- 1 ≤ `kernel_rows`, `kernel_cols` ≤ 31
- `kernel_rows` ≤ `input_rows`
- `kernel_cols` ≤ `input_cols`
- Performance is measured with `input_cols` = 3,072, `input_rows` = 3,072, `kernel_cols` = 15, `kernel_rows` = 15

Run `python3 puzzles/leetgpu/10_2d_convolution.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, OutTensor, RandTensor, run_challenge


class Challenge(ChallengeBase):
    name = "2D Convolution"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        input: torch.Tensor,
        kernel: torch.Tensor,
        output: torch.Tensor,
        input_rows: int,
        input_cols: int,
        kernel_rows: int,
        kernel_cols: int,
    ):
        # Reshape flattened arrays to 2D matrices
        input_2d = input.view(input_rows, input_cols)
        kernel_2d = kernel.view(kernel_rows, kernel_cols)
        # Prepare tensors for conv2d (add batch and channel dimensions)
        kernel_prepared = kernel_2d.unsqueeze(0).unsqueeze(0)
        input_prepared = input_2d.unsqueeze(0).unsqueeze(0)
        # Perform cross-correlation using PyTorch's F.conv2d
        # (which does cross-correlation by default)
        result = torch.nn.functional.conv2d(input_prepared, kernel_prepared, padding=0)
        # Copy result to output tensor (removing the extra dimensions and flattening)
        output.copy_(result.view(-1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "kernel": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "input_rows": (ctypes.c_int, "in"),
            "input_cols": (ctypes.c_int, "in"),
            "kernel_rows": (ctypes.c_int, "in"),
            "kernel_cols": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device=self.device, dtype=dtype
        )
        kernel = torch.tensor([0.0, 1.0, 1.0, 0.0], device=self.device, dtype=dtype)
        output = torch.empty(4, device=self.device, dtype=dtype)
        return {
            "input": input,
            "kernel": kernel,
            "output": output,
            "input_rows": 3,
            "input_cols": 3,
            "kernel_rows": 2,
            "kernel_cols": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # basic_example
        tests.append(
            {
                "input": torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device=self.device, dtype=dtype
                ),
                "kernel": torch.tensor([0.0, 1.0, 1.0, 0.0], device=self.device, dtype=dtype),
                "output": torch.empty(4, device=self.device, dtype=dtype),
                "input_rows": 3,
                "input_cols": 3,
                "kernel_rows": 2,
                "kernel_cols": 2,
            }
        )
        # rectangular_input
        tests.append(
            {
                "input": torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=self.device, dtype=dtype
                ),
                "kernel": torch.tensor([1.0, 0.0], device=self.device, dtype=dtype),
                "output": torch.empty(4, device=self.device, dtype=dtype),
                "input_rows": 2,
                "input_cols": 3,
                "kernel_rows": 1,
                "kernel_cols": 2,
            }
        )
        # negative_kernel
        tests.append(
            {
                "input": torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device=self.device, dtype=dtype
                ),
                "kernel": torch.tensor([-1.0, 1.0, 0.0, 0.0], device=self.device, dtype=dtype),
                "output": torch.empty(4, device=self.device, dtype=dtype),
                "input_rows": 3,
                "input_cols": 3,
                "kernel_rows": 2,
                "kernel_cols": 2,
            }
        )
        # single_element_kernel
        tests.append(
            {
                "input": torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device=self.device, dtype=dtype
                ),
                "kernel": torch.tensor([2.0], device=self.device, dtype=dtype),
                "output": torch.empty(9, device=self.device, dtype=dtype),
                "input_rows": 3,
                "input_cols": 3,
                "kernel_rows": 1,
                "kernel_cols": 1,
            }
        )
        # medium_matrix_small_kernel
        tests.append(
            {
                "input": torch.empty(64 * 64, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "kernel": torch.empty(3 * 3, device=self.device, dtype=dtype).uniform_(-0.5, 0.5),
                "output": torch.empty(62 * 62, device=self.device, dtype=dtype),
                "input_rows": 64,
                "input_cols": 64,
                "kernel_rows": 3,
                "kernel_cols": 3,
            }
        )
        # large_matrix_medium_kernel
        tests.append(
            {
                "input": torch.empty(128 * 128, device=self.device, dtype=dtype).uniform_(
                    -2.0, 2.0
                ),
                "kernel": torch.empty(7 * 7, device=self.device, dtype=dtype).uniform_(-0.2, 0.2),
                "output": torch.empty(122 * 122, device=self.device, dtype=dtype),
                "input_rows": 128,
                "input_cols": 128,
                "kernel_rows": 7,
                "kernel_cols": 7,
            }
        )
        # rectangular_large_matrix
        tests.append(
            {
                "input": torch.empty(128 * 256, device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "kernel": torch.empty(5 * 5, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "output": torch.empty(124 * 252, device=self.device, dtype=dtype),
                "input_rows": 128,
                "input_cols": 256,
                "kernel_rows": 5,
                "kernel_cols": 5,
            }
        )
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        input_rows = 3072
        input_cols = 3072
        kernel_rows = 15
        kernel_cols = 15
        input = RandTensor((input_rows * input_cols,), -1.0, 1.0)
        kernel = RandTensor((kernel_rows * kernel_cols,), -1.0, 1.0)
        output_rows = input_rows - kernel_rows + 1
        output_cols = input_cols - kernel_cols + 1
        output = OutTensor((output_rows * output_cols,))
        return {
            "input": input,
            "kernel": kernel,
            "output": output,
            "input_rows": input_rows,
            "input_cols": input_cols,
            "kernel_rows": kernel_rows,
            "kernel_cols": kernel_cols,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_2d_convolution(...):
#     ...


def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
