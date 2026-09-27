r"""
LeetGPU 16: Prefix Sum
======================

Category: ["leetgpu"]
Difficulty: ["medium"]

Write a GPU program that computes the prefix sum (cumulative sum) of an array of 32-bit floating point numbers.
For an input array `[a, b, c, d, ...]`, the prefix sum is `[a, a+b, a+b+c, a+b+c+d, ...]`.

(diagram omitted, see challenge.html in leetgpu-challenges)

Implementation Requirements
---------------------------

- Use only GPU native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The result must be stored in the `output` array

Example 1:
----------

Input: [1.0, 2.0, 3.0, 4.0]
Output: [1.0, 3.0, 6.0, 10.0]

Example 2:
----------

Input: [5.0, -2.0, 3.0, 1.0, -4.0]
Output: [5.0, 3.0, 6.0, 7.0, 3.0]

Constraints
-----------

- 1 ≤ `N` ≤ 100,000,000
- -1000.0 ≤ `input[i]` ≤ 1000.0
- The largest value in the output array will fit within a 32-bit float
- Performance is measured with `N` = 250,000

Run `python3 puzzles/leetgpu/16_prefix_sum.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Prefix Sum"
    atol = 0.01
    rtol = 0.01
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
        assert input.shape == (N,)
        assert output.shape == (N,)
        result = torch.cumsum(input, dim=0)
        output.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device, dtype=dtype)
        output = torch.empty(4, device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # basic_example
        tests.append(
            {
                "input": torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device, dtype=dtype),
                "output": torch.empty(4, device=self.device, dtype=dtype),
                "N": 4,
            }
        )
        # mixed_signs
        tests.append(
            {
                "input": torch.tensor([5.0, -2.0, 3.0, 1.0, -4.0], device=self.device, dtype=dtype),
                "output": torch.empty(5, device=self.device, dtype=dtype),
                "N": 5,
            }
        )
        # single_element
        tests.append(
            {
                "input": torch.tensor([42.0], device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 1,
            }
        )
        # power_of_two
        tests.append(
            {
                "input": torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=self.device, dtype=dtype
                ),
                "output": torch.empty(8, device=self.device, dtype=dtype),
                "N": 8,
            }
        )
        # all_zeros
        tests.append(
            {
                "input": torch.empty(1024, device=self.device, dtype=dtype).zero_(),
                "output": torch.empty(1024, device=self.device, dtype=dtype),
                "N": 1024,
            }
        )
        # random_large
        tests.append(
            {
                "input": torch.empty(2025, device=self.device, dtype=dtype).uniform_(-10.0, 10.0),
                "output": torch.empty(2025, device=self.device, dtype=dtype),
                "N": 2025,
            }
        )
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 250000
        input = torch.empty(N, device=self.device, dtype=dtype).uniform_(-100.0, 100.0)
        output = torch.empty(N, device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": N,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_prefix_sum(...):
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
