r"""
LeetGPU 48: 2D Subarray Sum
===========================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a program that computes the sum of a 2D subarray of 32-bit integers.
You are given an input 2D array `input` of length `N x M`, and two row indices `S_ROW` and `E_ROW` and two column indices `S_COL` and `E_COL`.
`S_ROW`, `E_ROW`, `S_COL` and `E_COL` are inclusive, 0-based start and end indices — compute the sum of `input[S_ROW..E_ROW][S_COL..E_COL]`.

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` variable

Example 1:
----------

Input:  input = [[1, 2, 3],
[4, 5, 1]]

S_ROW = 0, E_ROW = 1, S_COL = 1, E_COL = 2
Output: output = 11

Example 2:
----------

Input:  input = [[5, 10],
[5, 2]]
S_ROW = 0, E_ROW = 0, S_COL = 1, E_COL = 1
Output: output = 10

Constraints
-----------

- 1 ≤ `N, M` ≤ 10,000
- 1 ≤ `input[i]` ≤ 10
- 0 ≤ `S_ROW` ≤ `E_ROW` ≤ `N - 1`
- 0 ≤ `S_COL` ≤ `E_COL` ≤ `M - 1`
- Performance is measured with `M` = 10,000, `N` = 10,000

Run `python3 puzzles/leetgpu/48_2d_subarray_sum.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "2D Subarray Sum"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        N: int,
        M: int,
        S_ROW: int,
        E_ROW: int,
        S_COL: int,
        E_COL: int,
    ):
        # Validate input types and shapes
        assert input.shape == (N, M)
        assert output.shape == (1,)
        assert input.dtype == torch.int32
        assert output.dtype == torch.int32

        # add all elements of subarray (input[S_ROW..E_ROW][S_COL..E_COL])
        output[0] = torch.sum(input[S_ROW : E_ROW + 1, S_COL : E_COL + 1])

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
            "M": (ctypes.c_int, "in"),
            "S_ROW": (ctypes.c_int, "in"),
            "E_ROW": (ctypes.c_int, "in"),
            "S_COL": (ctypes.c_int, "in"),
            "E_COL": (ctypes.c_int, "in"),
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
            "S_ROW": 0,
            "E_ROW": 1,
            "S_COL": 1,
            "E_COL": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append(
            {
                "input": torch.tensor([[5, 10], [5, 2]], device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 2,
                "M": 2,
                "S_ROW": 0,
                "E_ROW": 0,
                "S_COL": 1,
                "E_COL": 1,
            }
        )

        # all_same_value
        tests.append(
            {
                "input": torch.tensor([[2] * 16] * 3, device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 3,
                "M": 16,
                "S_ROW": 0,
                "E_ROW": 2,
                "S_COL": 0,
                "E_COL": 15,
            }
        )

        # increasing_sequence
        tests.append(
            {
                "input": torch.randint(1, 11, (50, 50), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 50,
                "M": 50,
                "S_ROW": 0,
                "E_ROW": 49,
                "S_COL": 0,
                "E_COL": 49,
            }
        )

        # medium_size
        tests.append(
            {
                "input": torch.randint(1, 11, (100, 100), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 100,
                "M": 100,
                "S_ROW": 0,
                "E_ROW": 79,
                "S_COL": 1,
                "E_COL": 87,
            }
        )

        # large_size
        tests.append(
            {
                "input": torch.randint(1, 11, (1000, 1000), device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 1000,
                "M": 1000,
                "S_ROW": 10,
                "E_ROW": 951,
                "S_COL": 12,
                "E_COL": 810,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(1, 11, (10000, 10000), device=self.device, dtype=dtype)
        output = torch.empty(1, device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 10000,
            "M": 10000,
            "S_ROW": 0,
            "E_ROW": 9998,
            "S_COL": 1,
            "E_COL": 9999,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_2d_subarray_sum(...):
#     ...


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    M: int,
    S_ROW: int,
    E_ROW: int,
    S_COL: int,
    E_COL: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
