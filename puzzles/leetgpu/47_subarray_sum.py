r"""
LeetGPU 47: Subarray Sum
========================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a program that computes the sum of a subarray of 32-bit integers.
You are given an input array `input` of length `N`, and two indices `S` and `E`.
`S` and `E` are inclusive, 0-based start and end indices — compute the sum of `input[S..E]`.

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` variable

Example 1:
----------

Input: input = [1, 2, 1, 3, 4], S = 1, E = 3
Output: output = 6

Example 2:
----------

Input: input = [1, 2, 3, 4], S = 0, E = 3
Output: output = 10

Constraints
-----------

- 1 ≤ `N` ≤ 100,000,000
- 1 ≤ `input[i]` ≤ 10
- 0 ≤ `S` ≤ `E` ≤ `N - 1`
- Performance is measured with `N` = 100,000,000

Run `python3 puzzles/leetgpu/47_subarray_sum.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Subarray Sum"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, S: int, E: int):
        # Validate input types and shapes
        assert input.shape == (N,)
        assert output.shape == (1,)
        assert input.dtype == torch.int32
        assert output.dtype == torch.int32

        # add all element of subarray (input[S], ..., input[E])
        output[0] = torch.sum(input[S : E + 1])

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
            "E": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.tensor([1, 2, 1, 3, 4], device=self.device, dtype=dtype)
        output = torch.empty(1, device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 5,
            "S": 1,
            "E": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append(
            {
                "input": torch.tensor([1, 2, 3, 4], device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 4,
                "S": 0,
                "E": 3,
            }
        )

        # all_same_value
        tests.append(
            {
                "input": torch.tensor([2] * 16, device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 16,
                "S": 0,
                "E": 15,
            }
        )

        # increasing_sequence
        tests.append(
            {
                "input": torch.randint(1, 5, (32,), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 32,
                "S": 0,
                "E": 31,
            }
        )

        # medium_size
        tests.append(
            {
                "input": torch.randint(1, 10, (1000,), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 1000,
                "S": 0,
                "E": 500,
            }
        )

        # large_size
        tests.append(
            {
                "input": torch.randint(1, 11, (100000,), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 100000,
                "S": 123,
                "E": 98765,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(1, 11, (100000000,), device=self.device, dtype=dtype)
        output = torch.empty(1, device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 100000000,
            "S": 17651,
            "E": 98765431,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_subarray_sum(...):
#     ...


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    S: int,
    E: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
