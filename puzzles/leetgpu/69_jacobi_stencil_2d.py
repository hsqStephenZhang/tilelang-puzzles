r"""
LeetGPU 69: 2D Jacobi Stencil
=============================

Category: ["leetgpu"]
Difficulty: ["medium"]

Given a 2D grid of 32-bit floating point values, apply one iteration of the 5-point Jacobi stencil:
each interior cell of the output is set to the average of its four cardinal neighbors (top, bottom,
left, right) from the input grid. Boundary cells (first/last row and column) are copied unchanged
from the input to the output.

(diagram omitted, see challenge.html in leetgpu-challenges)

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in `output`
- Read exclusively from `input` and write exclusively to `output` (do not update `input`)

Example:
--------

Input (\(4 \times 4\)):
\[
\begin{bmatrix}
1.0 & 2.0 & 3.0 & 4.0 \\
5.0 & 6.0 & 7.0 & 8.0 \\
9.0 & 10.0 & 11.0 & 12.0 \\
13.0 & 14.0 & 15.0 & 16.0
\end{bmatrix}
\]
Output (\(4 \times 4\)):
\[
\begin{bmatrix}
1.0 & 2.0 & 3.0 & 4.0 \\
5.0 & 6.0 & 7.0 & 8.0 \\
9.0 & 10.0 & 11.0 & 12.0 \\
13.0 & 14.0 & 15.0 & 16.0
\end{bmatrix}
\]
Interior cell \((1,1)\): \(0.25 \times (\text{input}[0,1] + \text{input}[2,1] + \text{input}[1,0] + \text{input}[1,2])\)
\(= 0.25 \times (2.0 + 10.0 + 5.0 + 7.0) = 6.0\)

Interior cell \((1,2)\): \(0.25 \times (\text{input}[0,2] + \text{input}[2,2] + \text{input}[1,1] + \text{input}[1,3])\)
\(= 0.25 \times (3.0 + 11.0 + 6.0 + 8.0) = 7.0\)

Interior cell \((2,1)\): \(0.25 \times (\text{input}[1,1] + \text{input}[3,1] + \text{input}[2,0] + \text{input}[2,2])\)
\(= 0.25 \times (6.0 + 14.0 + 9.0 + 11.0) = 10.0\)

Interior cell \((2,2)\): \(0.25 \times (\text{input}[1,2] + \text{input}[3,2] + \text{input}[2,1] + \text{input}[2,3])\)
\(= 0.25 \times (7.0 + 15.0 + 10.0 + 12.0) = 11.0\)

Constraints
-----------

- 1 ≤ `rows`, `cols` ≤ 16,384
- Input values are in the range [-100, 100]
- All values are 32-bit floats
- Performance is measured with `rows` = 8,192, `cols` = 8,192

Run `python3 puzzles/leetgpu/69_jacobi_stencil_2d.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "2D Jacobi Stencil"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        rows: int,
        cols: int,
    ):
        assert input.shape == (rows, cols)
        assert output.shape == (rows, cols)
        assert input.dtype == torch.float32

        # Copy boundary cells unchanged
        output.copy_(input)

        # Apply 5-point stencil to interior cells:
        # output[i, j] = 0.25 * (input[i-1,j] + input[i+1,j] + input[i,j-1] + input[i,j+1])
        output[1:-1, 1:-1] = 0.25 * (
            input[0:-2, 1:-1]  # top neighbor
            + input[2:, 1:-1]  # bottom neighbor
            + input[1:-1, 0:-2]  # left neighbor
            + input[1:-1, 2:]  # right neighbor
        )

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "rows": (ctypes.c_int, "in"),
            "cols": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        output = torch.empty((4, 4), device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "rows": 4,
            "cols": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # minimal_3x3 (only one interior cell)
        tests.append(
            {
                "input": torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "output": torch.empty((3, 3), device=self.device, dtype=dtype),
                "rows": 3,
                "cols": 3,
            }
        )

        # minimal_1x1 (all boundary, no interior cells)
        tests.append(
            {
                "input": torch.tensor([[42.0]], device=self.device, dtype=dtype),
                "output": torch.empty((1, 1), device=self.device, dtype=dtype),
                "rows": 1,
                "cols": 1,
            }
        )

        # single_row (all boundary)
        tests.append(
            {
                "input": torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=self.device, dtype=dtype),
                "output": torch.empty((1, 4), device=self.device, dtype=dtype),
                "rows": 1,
                "cols": 4,
            }
        )

        # single_col (all boundary)
        tests.append(
            {
                "input": torch.tensor(
                    [[1.0], [2.0], [3.0], [4.0]], device=self.device, dtype=dtype
                ),
                "output": torch.empty((4, 1), device=self.device, dtype=dtype),
                "rows": 4,
                "cols": 1,
            }
        )

        # all_zeros (interior should stay zero)
        tests.append(
            {
                "input": torch.zeros((16, 16), device=self.device, dtype=dtype),
                "output": torch.empty((16, 16), device=self.device, dtype=dtype),
                "rows": 16,
                "cols": 16,
            }
        )

        # uniform_constant (interior stays the same when all values equal)
        tests.append(
            {
                "input": torch.full((32, 32), 3.14, device=self.device, dtype=dtype),
                "output": torch.empty((32, 32), device=self.device, dtype=dtype),
                "rows": 32,
                "cols": 32,
            }
        )

        # power_of_2_square_64
        tests.append(
            {
                "input": torch.empty((64, 64), device=self.device, dtype=dtype).uniform_(-5.0, 5.0),
                "output": torch.empty((64, 64), device=self.device, dtype=dtype),
                "rows": 64,
                "cols": 64,
            }
        )

        # power_of_2_square_128
        tests.append(
            {
                "input": torch.empty((128, 128), device=self.device, dtype=dtype).uniform_(
                    -10.0, 10.0
                ),
                "output": torch.empty((128, 128), device=self.device, dtype=dtype),
                "rows": 128,
                "cols": 128,
            }
        )

        # non_power_of_2_30x30
        tests.append(
            {
                "input": torch.empty((30, 30), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty((30, 30), device=self.device, dtype=dtype),
                "rows": 30,
                "cols": 30,
            }
        )

        # non_power_of_2_100x100
        tests.append(
            {
                "input": torch.empty((100, 100), device=self.device, dtype=dtype).uniform_(
                    -3.0, 3.0
                ),
                "output": torch.empty((100, 100), device=self.device, dtype=dtype),
                "rows": 100,
                "cols": 100,
            }
        )

        # non_square_255x33
        tests.append(
            {
                "input": torch.empty((255, 33), device=self.device, dtype=dtype).uniform_(
                    -2.0, 2.0
                ),
                "output": torch.empty((255, 33), device=self.device, dtype=dtype),
                "rows": 255,
                "cols": 33,
            }
        )

        # negative_values_non_square_17x97
        tests.append(
            {
                "input": torch.empty((17, 97), device=self.device, dtype=dtype).uniform_(
                    -100.0, 0.0
                ),
                "output": torch.empty((17, 97), device=self.device, dtype=dtype),
                "rows": 17,
                "cols": 97,
            }
        )

        # realistic_medium_512x256
        tests.append(
            {
                "input": torch.empty((512, 256), device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "output": torch.empty((512, 256), device=self.device, dtype=dtype),
                "rows": 512,
                "cols": 256,
            }
        )

        # realistic_large_1024x1024
        tests.append(
            {
                "input": torch.empty((1024, 1024), device=self.device, dtype=dtype).uniform_(
                    -5.0, 5.0
                ),
                "output": torch.empty((1024, 1024), device=self.device, dtype=dtype),
                "rows": 1024,
                "cols": 1024,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        rows = 8192
        cols = 8192
        return {
            "input": torch.empty((rows, cols), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
            "output": torch.empty((rows, cols), device=self.device, dtype=dtype),
            "rows": rows,
            "cols": cols,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_jacobi_stencil_2d(...):
#     ...


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    cols: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
