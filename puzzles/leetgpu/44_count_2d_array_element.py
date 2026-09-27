r"""
LeetGPU 44: Count 2D Array Element
==================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Write a GPU program that counts the number of elements with the integer value k in an 2D array of 32-bit integers.
The program should count the number of elements with k in an 2D array.
You are given an input 2D array `input` of length `N x M` and integer `k`.

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` variable

Example 1:
----------

Input: input [[1, 2, 3],
[4, 5, 1]]
k = 1
Output: output = 2

Example 2:
----------

Input: input [[5, 10],
[5, 2]]
k = 1
Output: output = 0

Constraints
-----------

- 1 ≤ `N, M` ≤ 10,000
- 1 ≤ `input[i], k` ≤ 100
- Performance is measured with `K` = 1, `M` = 10,000, `N` = 10,000

Run `python3 puzzles/leetgpu/44_count_2d_array_element.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Count 2D Array Element"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int):
        # Validate input types and shapes
        assert input.shape == (N, M)
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
            "M": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.tensor([[1, 2, 3], [4, 5, 1]], device=self.device, dtype=dtype)
        output = torch.empty(1, device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 2,
            "M": 3,
            "K": 1,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append(
            {
                "input": torch.tensor([[1, 2, 3], [4, 5, 1]], device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 2,
                "M": 3,
                "K": 1,
            }
        )

        # all_same_value
        tests.append(
            {
                "input": torch.tensor([[2] * 16] * 3, device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 3,
                "M": 16,
                "K": 2,
            }
        )

        # increasing_sequence
        tests.append(
            {
                "input": torch.randint(1, 11, (50, 50), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 50,
                "M": 50,
                "K": 5,
            }
        )

        # medium_size
        tests.append(
            {
                "input": torch.randint(1, 101, (100, 100), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 100,
                "M": 100,
                "K": 51,
            }
        )

        # large_size
        tests.append(
            {
                "input": torch.randint(1, 11, (1000, 1000), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 1000,
                "M": 1000,
                "K": 1,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(1, 3, (10000, 10000), device=self.device, dtype=dtype)
        output = torch.empty(1, device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 10000,
            "M": 10000,
            "K": 1,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_count_2d_array_element(...):
#     ...


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    M: int,
    K: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
