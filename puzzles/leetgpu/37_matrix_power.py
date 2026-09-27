r"""
LeetGPU 37: Matrix Power
========================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a GPU program that raises a square matrix \(A\) of size \(N \times N\) to an integer power \(P\).

The `solve` function receives a flattened input matrix `input` (row-major order), an empty output matrix `output` of the same size, the dimension `N`, and the exponent `P`.

You must compute \(\text{output} = A^{P}\) where matrix multiplication is standard dense multiplication over 32-bit floating point numbers.

Implementation Requirements
---------------------------

- External libraries are not permitted.
- The `solve` function signature must remain unchanged.
- The final result must be written to the `output` array in row-major order.

Example 1:
----------

Input:
input  = [[1.0, 2.0],
[3.0, 4.0]]
N      = 2
P      = 3
Output:
output = [[37.0, 54.0],
[81.0, 118.0]]

Example 2:
----------

Input:
input  = [[1.0, 0.0, 2.0],
[0.0, 1.0, 0.0],
[3.0, 0.0, 0.0]]
N      = 3
P      = 2
Output:
output = [[7.0, 0.0, 2.0],
[0.0, 1.0, 0.0],
[3.0, 0.0, 6.0]]

Constraints
-----------

- \(1 \le N \le 1024\)
- \(1 \le P \le 20\)
- Elements of `input` satisfy \(-10.0 \le A_{ij} \le 10.0\)
- Performance is measured with `N` = 512

Run `python3 puzzles/leetgpu/37_matrix_power.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Matrix Power"
    atol = 0.0001
    rtol = 0.0001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, P: int):
        """
        Matrix power implementation using PyTorch.
        Raises an N x N matrix to integer power P.
        """
        assert input.dtype == torch.float32
        assert output.dtype == torch.float32
        assert input.shape == output.shape == (N * N,)
        assert P >= 1

        mat = input.view(N, N)
        result = torch.linalg.matrix_power(mat, P).float()
        output[:] = result.reshape(-1)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "P": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 2
        P = 3
        input_data = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], device=self.device, dtype=dtype
        ).flatten()
        output_data = torch.zeros((2, 2), device=self.device, dtype=dtype).flatten()

        return {
            "input": input_data,
            "output": output_data,
            "N": N,
            "P": P,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_cases = []

        # Test case 1: example 2x2 power 3
        test_cases.append(
            {
                "input": torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], device=self.device, dtype=dtype
                ).flatten(),
                "output": torch.zeros((2, 2), device=self.device, dtype=dtype).flatten(),
                "N": 2,
                "P": 3,
            }
        )

        # Test case 2: identity 3x3 power 5
        test_cases.append(
            {
                "input": torch.eye(3, device=self.device, dtype=dtype).flatten(),
                "output": torch.zeros((3, 3), device=self.device, dtype=dtype).flatten(),
                "N": 3,
                "P": 5,
            }
        )

        # Test case 3: random 5x5 power 2
        test_cases.append(
            {
                "input": torch.empty((5, 5), device=self.device, dtype=dtype)
                .uniform_(-5.0, 5.0)
                .flatten(),
                "output": torch.zeros((5, 5), device=self.device, dtype=dtype).flatten(),
                "N": 5,
                "P": 2,
            }
        )

        # Test case 4: random 16x16 power 3
        test_cases.append(
            {
                "input": torch.empty((16, 16), device=self.device, dtype=dtype)
                .uniform_(-1.0, 1.0)
                .flatten(),
                "output": torch.zeros((16, 16), device=self.device, dtype=dtype).flatten(),
                "N": 16,
                "P": 3,
            }
        )

        # Test case 5: random 8x8 power 4
        test_cases.append(
            {
                "input": torch.empty((8, 8), device=self.device, dtype=dtype)
                .uniform_(-10.0, 10.0)
                .flatten(),
                "output": torch.zeros((8, 8), device=self.device, dtype=dtype).flatten(),
                "N": 8,
                "P": 4,
            }
        )

        # Test case 6: random 10x10 power 1
        test_cases.append(
            {
                "input": torch.empty((10, 10), device=self.device, dtype=dtype)
                .uniform_(-2.0, 2.0)
                .flatten(),
                "output": torch.zeros((10, 10), device=self.device, dtype=dtype).flatten(),
                "N": 10,
                "P": 1,
            }
        )

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 512
        P = 3
        return {
            "input": torch.empty((N, N), device=self.device, dtype=dtype)
            .uniform_(-10.0, 10.0)
            .flatten(),
            "output": torch.zeros((N, N), device=self.device, dtype=dtype).flatten(),
            "N": N,
            "P": P,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_matrix_power(...):
#     ...


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    P: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
