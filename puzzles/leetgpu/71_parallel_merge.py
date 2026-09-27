r"""
LeetGPU 71: Parallel Merge
==========================

Category: ["leetgpu"]
Difficulty: ["medium"]

Given two sorted arrays `A` of length `M` and `B` of length
`N`, both containing 32-bit floating-point values in non-decreasing order, produce a
single sorted array `C` of length `M + N` containing all elements of
`A` and `B` in non-decreasing order.

Implementation Requirements
---------------------------

- Use only GPU native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final merged result must be stored in `C`

Example
-------

Input:
A = [1.0, 3.0, 5.0, 7.0],  M = 4
B = [2.0, 4.0, 6.0, 8.0],  N = 4

Output:
C = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

Input:
A = [-1.0, 1.0, 3.0],  M = 3
B = [2.0],             N = 1

Output:
C = [-1.0, 1.0, 2.0, 3.0]

Constraints
-----------

- 1 ≤ `M`, `N` ≤ 50,000,000
- `M + N` ≤ 50,000,000
- Both `A` and `B` are sorted in non-decreasing order
- Elements are 32-bit floats
- Performance is measured with `M` = 25,000,000, `N` = 25,000,000

Run `python3 puzzles/leetgpu/71_parallel_merge.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Parallel Merge"
    atol = 0.0
    rtol = 0.0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        M: int,
        N: int,
    ):
        assert A.shape == (M,), f"Expected A.shape=({M},), got {A.shape}"
        assert B.shape == (N,), f"Expected B.shape=({N},), got {B.shape}"
        assert C.shape == (M + N,), f"Expected C.shape=({M + N},), got {C.shape}"
        assert A.dtype == torch.float32
        assert B.dtype == torch.float32
        assert C.dtype == torch.float32

        merged, _ = torch.sort(torch.cat([A, B]))
        C.copy_(merged)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "C": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        A = torch.tensor([1.0, 3.0, 5.0, 7.0], device=self.device, dtype=dtype)
        B = torch.tensor([2.0, 4.0, 6.0, 8.0], device=self.device, dtype=dtype)
        M, N = 4, 4
        C = torch.empty(M + N, device=self.device, dtype=dtype)
        return {"A": A, "B": B, "C": C, "M": M, "N": N}

    def _make_test(self, M: int, N: int, lo: float = -10.0, hi: float = 10.0) -> Dict[str, Any]:
        dtype = torch.float32
        A, _ = torch.sort(torch.empty(M, device=self.device, dtype=dtype).uniform_(lo, hi))
        B, _ = torch.sort(torch.empty(N, device=self.device, dtype=dtype).uniform_(lo, hi))
        C = torch.empty(M + N, device=self.device, dtype=dtype)
        return {"A": A, "B": B, "C": C, "M": M, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # Edge cases — tiny sizes
        tests.append(
            {
                "A": torch.tensor([0.0], device=self.device, dtype=dtype),
                "B": torch.tensor([1.0], device=self.device, dtype=dtype),
                "C": torch.empty(2, device=self.device, dtype=dtype),
                "M": 1,
                "N": 1,
            }
        )
        tests.append(
            {
                "A": torch.tensor([2.0], device=self.device, dtype=dtype),
                "B": torch.tensor([-1.0, 1.0, 3.0], device=self.device, dtype=dtype),
                "C": torch.empty(4, device=self.device, dtype=dtype),
                "M": 1,
                "N": 3,
            }
        )
        tests.append(
            {
                "A": torch.tensor([-1.0, 1.0, 3.0], device=self.device, dtype=dtype),
                "B": torch.tensor([2.0], device=self.device, dtype=dtype),
                "C": torch.empty(4, device=self.device, dtype=dtype),
                "M": 3,
                "N": 1,
            }
        )
        # All zeros
        tests.append(
            {
                "A": torch.zeros(2, device=self.device, dtype=dtype),
                "B": torch.zeros(2, device=self.device, dtype=dtype),
                "C": torch.empty(4, device=self.device, dtype=dtype),
                "M": 2,
                "N": 2,
            }
        )

        # Power-of-2 sizes
        tests.append(self._make_test(16, 16))
        tests.append(self._make_test(32, 32, lo=-100.0, hi=0.0))  # all negative
        tests.append(self._make_test(64, 128))
        tests.append(self._make_test(512, 512))
        tests.append(self._make_test(1024, 1024))

        # Non-power-of-2 sizes
        tests.append(self._make_test(30, 33))
        tests.append(self._make_test(100, 77))
        tests.append(self._make_test(255, 127))

        # A entirely less than B (no interleaving needed)
        A_low, _ = torch.sort(
            torch.empty(256, device=self.device, dtype=dtype).uniform_(-20.0, -10.0)
        )
        B_high, _ = torch.sort(
            torch.empty(256, device=self.device, dtype=dtype).uniform_(10.0, 20.0)
        )
        tests.append(
            {
                "A": A_low,
                "B": B_high,
                "C": torch.empty(512, device=self.device, dtype=dtype),
                "M": 256,
                "N": 256,
            }
        )

        # Many duplicate values
        A_dup = torch.sort(torch.randint(0, 5, (128,), device=self.device).to(dtype=dtype)).values
        B_dup = torch.sort(torch.randint(0, 5, (128,), device=self.device).to(dtype=dtype)).values
        tests.append(
            {
                "A": A_dup,
                "B": B_dup,
                "C": torch.empty(256, device=self.device, dtype=dtype),
                "M": 128,
                "N": 128,
            }
        )

        # Realistic size
        tests.append(self._make_test(5000, 7000))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M = 25_000_000
        N = 25_000_000
        A, _ = torch.sort(torch.empty(M, device=self.device, dtype=dtype).uniform_(-1.0, 1.0))
        B, _ = torch.sort(torch.empty(N, device=self.device, dtype=dtype).uniform_(-1.0, 1.0))
        C = torch.empty(M + N, device=self.device, dtype=dtype)
        return {"A": A, "B": B, "C": C, "M": M, "N": N}


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_parallel_merge(...):
#     ...


def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
):
    # TODO: launch your TileLang kernel and write the result into `C` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
