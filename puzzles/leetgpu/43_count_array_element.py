r"""
LeetGPU 43: Count Array Element
===============================

Category: ["leetgpu"]
Difficulty: ["medium"]

Write a GPU program that counts the number of elements with the integer value k in an array of 32-bit integers.
The program should count the number of elements with k in an array.
You are given an input array `input` of length `N` and integer `k`.

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` variable

Example 1:
----------

Input: [1, 2, 3, 4, 1], k = 1
Output: 2

Example 2:
----------

Input: [5, 10, 5, 2], k = 11
Output: 0

Constraints
-----------

- 1 ≤ `N` ≤ 100,000,000
- 1 ≤ `input[i], k` ≤ 100,000
- Performance is measured with `K` = 501,010, `N` = 100,000,000

Run `python3 puzzles/leetgpu/43_count_array_element.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Count Array Element"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, K: int):
        # Validate input types and shapes
        assert input.shape == (N,)
        assert output.shape == (1,)
        assert input.dtype == torch.int32
        assert output.dtype == torch.int32

        # count the number of element with value k in an input array
        equality_tensor = input == K
        output[0] = torch.sum(equality_tensor)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.tensor([1, 2, 3, 4, 1], device=self.device, dtype=dtype)
        output = torch.empty(1, device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 5,
            "K": 1,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append(
            {
                "input": torch.tensor([1, 2, 3, 4, 1], device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 5,
                "K": 1,
            }
        )

        # all_same_value
        tests.append(
            {
                "input": torch.tensor([2] * 16, device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 16,
                "K": 2,
            }
        )

        # increasing_sequence
        tests.append(
            {
                "input": torch.randint(1, 5, (32,), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 32,
                "K": 4,
            }
        )

        # medium_size
        tests.append(
            {
                "input": torch.randint(1, 10, (1000,), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 1000,
                "K": 5,
            }
        )

        # large_size
        tests.append(
            {
                "input": torch.randint(1, 1000, (100000,), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 100000,
                "K": 501,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(1, 100001, (100000000,), device=self.device, dtype=dtype)
        output = torch.empty(1, device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 100000000,
            "K": 501010,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_count_array_element(...):
#     ...


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    K: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
