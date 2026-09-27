r"""
LeetGPU 51: Max Subarray Sum
============================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a program that computes the maximum sum of any contiguous subarray of length exactly `window_size`. You are given an array `input` of length `N` consisting of 32-bit signed integers, and an integer `window_size`.

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` variable

Example 1:
----------

Input:  input = [1, 2, 4, 2, 3], window_size = 2
Output: output = 6

Example 2:
----------

Input:  input = [-1, -4, -2, 1], window_size = 3
Output: output = -5

Constraints
-----------

- 1 ≤ `N` ≤ 50,000
- -10 ≤ `input[i]` ≤ 10
- 1 ≤ `window_size` ≤ `N`
- Performance is measured with `N` = 50,000

Run `python3 puzzles/leetgpu/51_max_subarray_sum.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Max Subarray Sum"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, window_size: int):
        # Validate input types and shapes
        assert input.shape == (N,)
        assert output.shape == (1,)
        assert input.dtype == torch.int32
        assert output.dtype == torch.int32

        # Computes the maximum sum of any contiguous subarray of length exactly window_size
        # using a sliding window approach.

        # Compute the sum of the first window_size elements (the initial window)
        current_sum = input[:window_size].sum()

        # Initialize max_sum with the sum of the first window
        max_sum = current_sum

        # Slide the window across the array from index window_size to N - 1
        for i in range(window_size, N):
            # Update the current sum by subtracting the element leaving the window
            # and adding the new element entering the window
            current_sum += input[i] - input[i - window_size]

            # Update max_sum if the current sum is greater
            max_sum = torch.max(max_sum, current_sum)

        # Store the final result in the output tensor
        output[0] = max_sum

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
            "window_size": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.tensor([1, 2, 4, 2, 3], device=self.device, dtype=dtype)
        output = torch.empty(1, device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 5,
            "window_size": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append(
            {
                "input": torch.tensor([-1, -4, -2, 1], device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 4,
                "window_size": 3,
            }
        )

        # all_same_value
        tests.append(
            {
                "input": torch.tensor([2] * 16, device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 16,
                "window_size": 15,
            }
        )

        # all_minus_value
        tests.append(
            {
                "input": torch.tensor([-10] * 1000, device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 1000,
                "window_size": 500,
            }
        )

        # increasing_sequence
        tests.append(
            {
                "input": torch.randint(-10, 11, (123,), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 123,
                "window_size": 7,
            }
        )

        # medium_size
        tests.append(
            {
                "input": torch.randint(-10, 11, (1000,), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 1000,
                "window_size": 476,
            }
        )

        # large_size
        tests.append(
            {
                "input": torch.randint(-10, 11, (10000,), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 10000,
                "window_size": 7011,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(-10, 11, (50000,), device=self.device, dtype=dtype)
        output = torch.empty(1, device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 50000,
            "window_size": 25000,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_max_subarray_sum(...):
#     ...


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    window_size: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
