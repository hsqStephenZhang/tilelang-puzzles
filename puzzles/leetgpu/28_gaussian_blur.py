r"""
LeetGPU 28: Gaussian Blur
=========================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a program that applies a Gaussian blur filter to a 2D image. Given an input image represented as a floating-point array and a Gaussian kernel, the program should compute the convolution of the image with the kernel.
All inputs and outputs are stored in row-major order.

The Gaussian blur is performed by convolving each pixel with a weighted average of its neighbors, where the weights are determined by the Gaussian kernel. For each output pixel at position (i, j), the value is calculated as:

\[ output[i, j] = \sum_{m=-k_h/2}^{k_h/2} \sum_{n=-k_w/2}^{k_w/2} input[i+m, j+n] \times kernel[m+k_h/2, n+k_w/2] \]

where \(k_h\) and \(k_w\) are the kernel height and width.

Implementation Requirements
---------------------------

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` array
- Handle boundary conditions by using zero-padding (treat values outside the image boundary as zeros)

Example 1:
----------

Input:
image (5, 5) = [
[1.0, 2.0, 3.0, 4.0, 5.0],
[6.0, 7.0, 8.0, 9.0, 10.0],
[11.0, 12.0, 13.0, 14.0, 15.0],
[16.0, 17.0, 18.0, 19.0, 20.0],
[21.0, 22.0, 23.0, 24.0, 25.0]
]

kernel (3, 3) = [
[0.0625, 0.125, 0.0625],
[0.125, 0.25, 0.125],
[0.0625, 0.125, 0.0625]
]

Output:
output (5, 5) = [
[1.6875, 2.75, 3.5, 4.25, 3.5625],
[4.75, 7.0, 8.0, 9.0, 7.25],
[8.5, 12.0, 13.0, 14.0, 11.0],
[12.25, 17.0, 18.0, 19.0, 14.75],
[11.0625, 15.25, 16.0, 16.75, 12.9375]
]

Example 2:
----------

Input:
image (3, 3) = [
[10.0, 20.0, 30.0],
[40.0, 50.0, 60.0],
[70.0, 80.0, 90.0]
]

kernel (3, 3) = [
[0.1, 0.1, 0.1],
[0.1, 0.2, 0.1],
[0.1, 0.1, 0.1]
]

Output:
output (3, 3) = [
[13.0, 23.0, 19.0],
[31.0, 50.0, 39.0],
[31.0, 47.0, 37.0]
]

Constraints
-----------

- 1 ≤ `input_rows`, `input_cols` ≤ 4096
- 3 ≤ `kernel_rows`, `kernel_cols` ≤ 21
- Both `kernel_rows` and `kernel_cols` will be odd numbers
- All kernel values will be non-negative and sum to 1.0 (normalized)
- Performance is measured with `input_cols` = 512, `input_rows` = 512, `kernel_cols` = 7, `kernel_rows` = 7

Run `python3 puzzles/leetgpu/28_gaussian_blur.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Gaussian Blur"
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
        input_2d = input.view(1, 1, input_rows, input_cols)
        kernel_2d = kernel.view(1, 1, kernel_rows, kernel_cols)
        pad_h = kernel_rows // 2
        pad_w = kernel_cols // 2
        result = torch.nn.functional.conv2d(input_2d, kernel_2d, padding=(pad_h, pad_w))
        output[:] = result.view(-1)

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
        input_rows, input_cols = 5, 5
        kernel_rows, kernel_cols = 3, 3
        input = torch.tensor(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
                25.0,
            ],
            device=self.device,
            dtype=dtype,
        )
        kernel = torch.tensor(
            [0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625],
            device=self.device,
            dtype=dtype,
        )
        output = torch.empty(input_rows * input_cols, device=self.device, dtype=dtype)
        return {
            "input": input,
            "kernel": kernel,
            "output": output,
            "input_rows": input_rows,
            "input_cols": input_cols,
            "kernel_rows": kernel_rows,
            "kernel_cols": kernel_cols,
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
                        [1.0, 2.0, 3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0, 9.0, 10.0],
                        [11.0, 12.0, 13.0, 14.0, 15.0],
                        [16.0, 17.0, 18.0, 19.0, 20.0],
                        [21.0, 22.0, 23.0, 24.0, 25.0],
                    ],
                    device=device,
                    dtype=dtype,
                ).flatten(),
                "kernel": torch.tensor(
                    [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]],
                    device=device,
                    dtype=dtype,
                ).flatten(),
                "output": torch.zeros((5, 5), device=device, dtype=dtype).flatten(),
                "input_rows": 5,
                "input_cols": 5,
                "kernel_rows": 3,
                "kernel_cols": 3,
            }
        )

        # identity_kernel
        tests.append(
            {
                "input": torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device=device, dtype=dtype
                ).flatten(),
                "kernel": torch.tensor(
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype
                ).flatten(),
                "output": torch.zeros((3, 3), device=device, dtype=dtype).flatten(),
                "input_rows": 3,
                "input_cols": 3,
                "kernel_rows": 3,
                "kernel_cols": 3,
            }
        )

        # all_ones_input
        tests.append(
            {
                "input": torch.ones((4, 4), device=device, dtype=dtype).flatten(),
                "kernel": torch.full((3, 3), 0.111111, device=device, dtype=dtype).flatten(),
                "output": torch.zeros((4, 4), device=device, dtype=dtype).flatten(),
                "input_rows": 4,
                "input_cols": 4,
                "kernel_rows": 3,
                "kernel_cols": 3,
            }
        )

        # single_pixel
        tests.append(
            {
                "input": torch.tensor([[42.0]], device=device, dtype=dtype).flatten(),
                "kernel": torch.tensor([[1.0]], device=device, dtype=dtype).flatten(),
                "output": torch.zeros((1, 1), device=device, dtype=dtype).flatten(),
                "input_rows": 1,
                "input_cols": 1,
                "kernel_rows": 1,
                "kernel_cols": 1,
            }
        )

        # large_random
        input_large = torch.empty((32, 32), device=device, dtype=dtype).uniform_(-10.0, 10.0)
        kernel_large = torch.empty((5, 5), device=device, dtype=dtype).uniform_(0.0, 1.0)
        tests.append(
            {
                "input": input_large.flatten(),
                "kernel": kernel_large.flatten(),
                "output": torch.zeros((32, 32), device=device, dtype=dtype).flatten(),
                "input_rows": 32,
                "input_cols": 32,
                "kernel_rows": 5,
                "kernel_cols": 5,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input_rows, input_cols = 512, 512
        kernel_rows, kernel_cols = 7, 7
        input = torch.empty(input_rows * input_cols, device=self.device, dtype=dtype).uniform_(
            0.0, 255.0
        )
        kernel = torch.empty(kernel_rows * kernel_cols, device=self.device, dtype=dtype).uniform_(
            0.0001, 0.02
        )
        output = torch.empty(input_rows * input_cols, device=self.device, dtype=dtype)
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
# def tl_gaussian_blur(...):
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
