r"""
LeetGPU 17: Dot Product
=======================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a GPU program that computes the dot product of two vectors containing 32-bit floating point numbers.
The dot product is the sum of the products of the corresponding elements of two vectors.

Mathematically, the dot product of two vectors \(A\) and \(B\) of length \(n\) is defined as:
\[
A \cdot B = \sum_{i=0}^{n-1} A_i \cdot B_i = A_0 \cdot B_0 + A_1 \cdot B_1 + \ldots + A_{n-1} \cdot B_{n-1}
\]

Implementation Requirements
---------------------------

- Use only GPU native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the output variable

Example 1:
----------

Input:  A = [1.0, 2.0, 3.0, 4.0]
B = [5.0, 6.0, 7.0, 8.0]
Output: result = 70.0  (1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0)

Example 2:
----------

Input:  A = [0.5, 1.5, 2.5]
B = [2.0, 3.0, 4.0]
Output: result = 15.5  (0.5*2.0 + 1.5*3.0 + 2.5*4.0)

Constraints
-----------

- `A` and `B` have identical lengths
- 1 ≤ `N` ≤ 100,000,000
- Performance is measured with `N` = 5

Run `python3 puzzles/leetgpu/17_dot_product.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Dot Product"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
        assert A.shape == (N,)
        assert B.shape == (N,)
        assert result.shape == (1,)
        result[0] = torch.dot(A, B)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "result": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        A = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device, dtype=dtype)
        B = torch.tensor([5.0, 6.0, 7.0, 8.0], device=self.device, dtype=dtype)
        result = torch.empty(1, device=self.device, dtype=dtype)
        return {
            "A": A,
            "B": B,
            "result": result,
            "N": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # basic_small
        tests.append(
            {
                "A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device, dtype=dtype),
                "B": torch.tensor([5.0, 6.0, 7.0, 8.0], device=self.device, dtype=dtype),
                "result": torch.empty(1, device=self.device, dtype=dtype),
                "N": 4,
            }
        )
        # all_zeros
        tests.append(
            {
                "A": torch.tensor([0.0] * 16, device=self.device, dtype=dtype),
                "B": torch.tensor([0.0] * 16, device=self.device, dtype=dtype),
                "result": torch.empty(1, device=self.device, dtype=dtype),
                "N": 16,
            }
        )
        # negative_numbers
        tests.append(
            {
                "A": torch.tensor([-1.0, -2.0, -3.0, -4.0], device=self.device, dtype=dtype),
                "B": torch.tensor([-5.0, -6.0, -7.0, -8.0], device=self.device, dtype=dtype),
                "result": torch.empty(1, device=self.device, dtype=dtype),
                "N": 4,
            }
        )
        # mixed_positive_negative
        tests.append(
            {
                "A": torch.tensor([1.0, -2.0, 3.0, -4.0], device=self.device, dtype=dtype),
                "B": torch.tensor([-1.0, 2.0, -3.0, 4.0], device=self.device, dtype=dtype),
                "result": torch.empty(1, device=self.device, dtype=dtype),
                "N": 4,
            }
        )
        # orthogonal_vectors
        tests.append(
            {
                "A": torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=dtype),
                "B": torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=dtype),
                "result": torch.empty(1, device=self.device, dtype=dtype),
                "N": 3,
            }
        )
        # medium_sized_vector
        tests.append(
            {
                "A": torch.empty(1000, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "B": torch.empty(1000, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "result": torch.empty(1, device=self.device, dtype=dtype),
                "N": 1000,
            }
        )
        # large_vector
        tests.append(
            {
                "A": torch.empty(10000, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "B": torch.empty(10000, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "result": torch.empty(1, device=self.device, dtype=dtype),
                "N": 10000,
            }
        )
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 5
        A = torch.empty(N, device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        B = torch.empty(N, device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        result = torch.zeros(1, device=self.device, dtype=dtype)
        return {
            "A": A,
            "B": B,
            "result": result,
            "N": N,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_dot_product(...):
#     ...


def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    result: torch.Tensor,
    N: int,
):
    # TODO: launch your TileLang kernel and write the result into `result` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
