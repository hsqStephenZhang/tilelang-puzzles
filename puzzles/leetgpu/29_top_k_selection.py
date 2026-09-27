r"""
LeetGPU 29: Top K Selection
===========================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a GPU program that, given a 1D array `input` of 32-bit floating point numbers of length `N`, selects the `k` largest elements and writes them in descending order to the `output` array of length `k`.

Implementation Requirements
---------------------------

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` array

Example 1:
----------

Input:
input = [1.0, 5.0, 3.0, 2.0, 4.0]
N = 5
k = 3

Output:
output = [5.0, 4.0, 3.0]

Example 2:
----------

Input:
input = [7.2, -1.0, 3.3, 8.8, 2.2]
N = 5
k = 2

Output:
output = [8.8, 7.2]

Constraints
-----------

- 1 ≤ N ≤ 100,000,000
- 1 ≤ k ≤ N
- All values in `input` are 32-bit floats
- Performance is measured with `N` = 50,000,000, `k` = 100

Run `python3 puzzles/leetgpu/29_top_k_selection.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Top K Selection"
    atol = 1e-05
    rtol = 0.0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, k: int):
        assert input.shape == (N,)
        assert output.shape == (k,)
        assert input.dtype == output.dtype == torch.float32
        assert input.device == output.device
        topk = torch.topk(input, k, largest=True).values
        output.copy_(topk)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "k": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0], device=self.device, dtype=dtype)
        output = torch.empty(3, device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 5,
            "k": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # basic_example
        tests.append(
            {
                "input": torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0], device=self.device, dtype=dtype),
                "output": torch.empty(3, device=self.device, dtype=dtype),
                "N": 5,
                "k": 3,
            }
        )
        # negative_numbers
        tests.append(
            {
                "input": torch.tensor(
                    [-2.0, -1.0, -3.0, -4.0, -5.0, -6.0], device=self.device, dtype=dtype
                ),
                "output": torch.empty(2, device=self.device, dtype=dtype),
                "N": 6,
                "k": 2,
            }
        )
        # all_equal
        tests.append(
            {
                "input": torch.tensor([7.0, 7.0, 7.0, 7.0], device=self.device, dtype=dtype),
                "output": torch.empty(3, device=self.device, dtype=dtype),
                "N": 4,
                "k": 3,
            }
        )
        # all_zeros with k == N
        tests.append(
            {
                "input": torch.zeros(4, device=self.device, dtype=dtype),
                "output": torch.empty(4, device=self.device, dtype=dtype),
                "N": 4,
                "k": 4,
            }
        )
        # single_element
        tests.append(
            {
                "input": torch.tensor([42.0], device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 1,
                "k": 1,
            }
        )
        # reverse_sorted
        tests.append(
            {
                "input": torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device=self.device, dtype=dtype),
                "output": torch.empty(2, device=self.device, dtype=dtype),
                "N": 5,
                "k": 2,
            }
        )
        # large_random (simulated; actual is random in runner)
        N, k = 1000, 10
        tests.append(
            {
                "input": torch.empty(N, device=self.device, dtype=dtype).uniform_(-1000.0, 1000.0),
                "output": torch.empty(k, device=self.device, dtype=dtype),
                "N": N,
                "k": k,
            }
        )
        # Multiple winners share a strided partition, or a contiguous region.
        # Keeping just one maximum per partition must not discard other winners.
        for N, k, stride, start in [
            (65536, 50, 1024, 0),
            (1048576, 100, 4096, 0),
            (50000000, 100, 1, 25000000),
        ]:
            input = torch.empty(N, device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
            # Integer-valued winners are exact in float32 and separated by much
            # more than the absolute comparison tolerance of 1e-5.
            input[start : start + k * stride : stride] = torch.arange(
                k + 100, 100, -1, device=self.device, dtype=dtype
            )
            tests.append(
                {
                    "input": input,
                    "output": torch.empty(k, device=self.device, dtype=dtype),
                    "N": N,
                    "k": k,
                }
            )
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 50000000
        k = 100
        return {
            "input": torch.empty(N, device=self.device, dtype=dtype).uniform_(-1e6, 1e6),
            "output": torch.empty(k, device=self.device, dtype=dtype),
            "N": N,
            "k": k,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_top_k_selection(...):
#     ...


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    k: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
