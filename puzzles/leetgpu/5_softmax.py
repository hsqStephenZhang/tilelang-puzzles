r"""
LeetGPU 5: Softmax
==================

Category: ["leetgpu"]
Difficulty: ["medium"]

Write a program that computes the softmax function for an array of 32-bit floating-point numbers on a GPU.  The softmax function is defined as follows:

For an input array \(x\) of length \(n\), the softmax of \(x\), denoted \(\sigma(x)\), is an array of length \(n\) where the \(i\)-th element is:

\(\sigma(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}\)

Your solution should handle potential overflow issues by using the "max trick".  Subtract the maximum value of the input array from each element before exponentiation.

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`

Example 1:
----------

Input: [1.0, 2.0, 3.0], N = 3
Output: [0.090, 0.244, 0.665] (approximately)

Example 2:
----------

Input: [-10.0, -5.0, 0.0, 5.0, 10.0], N = 5
Output: [2.047e-09, 3.038e-07, 4.509e-05, 6.693e-03, 9.933e-01] (approximately)

Constraints
-----------

- 1 ≤ `N` ≤ 500,000
- Performance is measured with `N` = 500,000

Run `python3 puzzles/leetgpu/5_softmax.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Softmax"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
        assert input.shape == output.shape == (N,)
        assert input.dtype == output.dtype
        assert input.device == output.device
        max_val = torch.max(input)
        exp_x = torch.exp(input - max_val)
        sum_exp = torch.sum(exp_x)
        output.copy_(exp_x / sum_exp)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input = torch.tensor([1.0, 2.0, 3.0], device=self.device, dtype=dtype)
        output = torch.empty(3, device=self.device, dtype=dtype)
        N = 3
        return {"input": input, "output": output, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # basic_small
        tests.append(
            {
                "input": torch.tensor([1.0, 2.0, 3.0], device=self.device, dtype=dtype),
                "output": torch.empty(3, device=self.device, dtype=dtype),
                "N": 3,
            }
        )
        # all_zeros
        tests.append(
            {
                "input": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype),
                "output": torch.empty(4, device=self.device, dtype=dtype),
                "N": 4,
            }
        )
        # negative_numbers
        tests.append(
            {
                "input": torch.tensor([-1.0, -2.0, -3.0], device=self.device, dtype=dtype),
                "output": torch.empty(3, device=self.device, dtype=dtype),
                "N": 3,
            }
        )
        # mixed_positive_negative
        tests.append(
            {
                "input": torch.tensor([1.0, -2.0, 3.0, -4.0], device=self.device, dtype=dtype),
                "output": torch.empty(4, device=self.device, dtype=dtype),
                "N": 4,
            }
        )
        # very_small_numbers
        tests.append(
            {
                "input": torch.tensor([1e-6, 1e-7, 1e-8, 1e-9], device=self.device, dtype=dtype),
                "output": torch.empty(4, device=self.device, dtype=dtype),
                "N": 4,
            }
        )
        # large_numbers
        tests.append(
            {
                "input": torch.tensor([10.0, 15.0, 20.0], device=self.device, dtype=dtype),
                "output": torch.empty(3, device=self.device, dtype=dtype),
                "N": 3,
            }
        )
        # single_element
        tests.append(
            {
                "input": torch.tensor([5.0], device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 1,
            }
        )
        # all_same_values
        tests.append(
            {
                "input": torch.tensor([2.5] * 10, device=self.device, dtype=dtype),
                "output": torch.empty(10, device=self.device, dtype=dtype),
                "N": 10,
            }
        )
        # large_array
        tests.append(
            {
                "input": torch.empty(2048, device=self.device, dtype=dtype).uniform_(0.0, 10.0),
                "output": torch.empty(2048, device=self.device, dtype=dtype),
                "N": 2048,
            }
        )
        # large_max_small_values
        tests.append(
            {
                "input": torch.tensor([1000.0, 1.0, 2.0, 3.0], device=self.device, dtype=dtype),
                "output": torch.empty(4, device=self.device, dtype=dtype),
                "N": 4,
            }
        )
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 500000
        input = torch.empty(N, device=self.device, dtype=dtype).uniform_(-10.0, 10.0)
        output = torch.empty(N, device=self.device, dtype=dtype)
        return {"input": input, "output": output, "N": N}


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_softmax(...):
#     ...


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
