r"""
LeetGPU 4: Reduction
====================

Category: ["leetgpu"]
Difficulty: ["medium"]

Write a GPU program that performs parallel reduction on an array of 32-bit floating point numbers to compute their sum.
The program should take an input array and produce a single output value containing the sum of all elements.

Implementation Requirements
---------------------------

- Use only GPU native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` variable

Example 1:
----------

Input: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Output: 36.0

Example 2:
----------

Input: [-2.5, 1.5, -1.0, 2.0]
Output: 0.0

Constraints
-----------

- 1 ≤ `N` ≤ 100,000,000
- -1000.0 ≤ `input[i]` ≤ 1000.0
- The final sum will always fit within a 32-bit float
- Performance is measured with `N` = 4,194,304

Run `python3 puzzles/leetgpu/4_reduction.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Reduction"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
        assert input.shape == (N,)
        assert output.shape == (1,)
        assert input.dtype == output.dtype
        assert input.device == output.device
        output[0] = torch.sum(input.double()).float()

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=self.device, dtype=dtype
        )
        output = torch.empty(1, device=self.device, dtype=dtype)
        N = 8
        return {"input": input, "output": output, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # basic_example
        tests.append(
            {
                "input": torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=self.device, dtype=dtype
                ),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 8,
            }
        )
        # negative_numbers
        tests.append(
            {
                "input": torch.tensor([-2.5, 1.5, -1.0, 2.0], device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 4,
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
        # all_zeros
        tests.append(
            {
                "input": torch.zeros(1024, device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 1024,
            }
        )
        # all_ones
        tests.append(
            {
                "input": torch.ones(1024, device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 1024,
            }
        )
        # non_power_of_two
        tests.append(
            {
                "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 5,
            }
        )
        # large_random
        tests.append(
            {
                "input": torch.empty(10000, device=self.device, dtype=dtype).uniform_(
                    -1000.0, 1000.0
                ),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 10000,
            }
        )
        # large_random_2
        tests.append(
            {
                "input": torch.empty(15000000, device=self.device, dtype=dtype).uniform_(
                    0.0, 1000.0
                ),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 15000000,
            }
        )
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 4_194_304
        input = torch.empty(N, device=self.device, dtype=dtype).uniform_(0.0, 1000.0)
        output = torch.empty(1, device=self.device, dtype=dtype)
        return {"input": input, "output": output, "N": N}


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_reduction(...):
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
