r"""
LeetGPU 13: Histogramming
=========================

Category: ["leetgpu"]
Difficulty: ["medium"]

Write a GPU program that computes the histogram of an array of 32-bit integers.
The histogram should count the number of occurrences of each integer value in the range `[0, num_bins)`.
You are given an input array `input` of length `N` and the number of bins `num_bins`.

The result should be an array of integers of length
`num_bins`, where each element represents
the count of occurrences of its corresponding index in the input array.

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The
`solve` function signature must remain unchanged

- The final result must be stored in the
`histogram` array.

Examples
--------

Input: input = [0, 1, 2, 1, 0],  N = 5, num_bins = 3
Output: [2, 2, 1]

Input: input = [3, 3, 3, 3], N = 4, num_bins = 5
Output: [0, 0, 0, 4, 0]

Constraints
-----------

- 1 ≤ `N` ≤ 100,000,000
- 0 ≤ `input[i]` < `num_bins`
- 1 ≤ `num_bins` ≤ 1024
- Performance is measured with `N` = 50,000,000, `num_bins` = 256

Run `python3 puzzles/leetgpu/13_histogramming.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Histogramming"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
        # Validate input types and shapes
        assert input.dtype == torch.int32
        assert histogram.dtype == torch.int32
        assert input.numel() == N
        assert histogram.numel() == num_bins
        # Zero out the histogram
        histogram.zero_()
        # Only count valid input values
        valid_mask = (input >= 0) & (input < num_bins)
        valid_input = input[valid_mask]
        counts = torch.bincount(valid_input, minlength=num_bins)
        histogram.copy_(counts)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_int), "in"),
            "histogram": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
            "num_bins": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.tensor([0, 1, 2, 1, 0], device=self.device, dtype=dtype)
        histogram = torch.empty(3, device=self.device, dtype=dtype)
        return {
            "input": input,
            "histogram": histogram,
            "N": 5,
            "num_bins": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append(
            {
                "input": torch.tensor([0, 1, 2, 1, 0], device=self.device, dtype=dtype),
                "histogram": torch.zeros(3, device=self.device, dtype=dtype),
                "N": 5,
                "num_bins": 3,
            }
        )

        # all_same_value
        tests.append(
            {
                "input": torch.tensor([2] * 16, device=self.device, dtype=dtype),
                "histogram": torch.zeros(5, device=self.device, dtype=dtype),
                "N": 16,
                "num_bins": 5,
            }
        )

        # increasing_sequence
        tests.append(
            {
                "input": torch.randint(0, 4, (32,), device=self.device, dtype=dtype),
                "histogram": torch.zeros(4, device=self.device, dtype=dtype),
                "N": 32,
                "num_bins": 4,
            }
        )

        # medium_size
        tests.append(
            {
                "input": torch.randint(0, 10, (1000,), device=self.device, dtype=dtype),
                "histogram": torch.zeros(10, device=self.device, dtype=dtype),
                "N": 1000,
                "num_bins": 10,
            }
        )

        # large_multi_block
        tests.append(
            {
                "input": torch.randint(0, 128, (10000,), device=self.device, dtype=dtype),
                "histogram": torch.zeros(128, device=self.device, dtype=dtype),
                "N": 10000,
                "num_bins": 128,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(0, 256, (50000000,), device=self.device, dtype=dtype)
        histogram = torch.zeros(256, device=self.device, dtype=dtype)
        return {
            "input": input,
            "histogram": histogram,
            "N": 50000000,
            "num_bins": 256,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_histogramming(...):
#     ...


def solve(
    input: torch.Tensor,
    histogram: torch.Tensor,
    N: int,
    num_bins: int,
):
    # TODO: launch your TileLang kernel and write the result into `histogram` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
