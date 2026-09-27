r"""
LeetGPU 72: Stream Compaction
=============================

Category: ["leetgpu"]
Difficulty: ["medium"]

Given a 1D array `A` of `N` 32-bit floating point numbers, compact all
positive elements (`A[i] > 0`) to the front of the output array `out`,
preserving their original relative order. Fill any remaining positions with `0.0`.
Stream compaction is a fundamental GPU primitive used throughout rendering, sparse computation,
and collision detection.

Implementation Requirements
---------------------------

- Use only native GPU features (external libraries are not permitted)
- The `solve` function signature must remain unchanged

-
The first k positions of `out` must contain the k elements of
`A` where `A[i] > 0`, in their original order

- Positions k through N−1 of `out` must be `0.0`
- Elements where `A[i] = 0.0` are not selected

Example
-------

Input:  A = [1.0, -2.0, 3.0, 0.0, -1.0, 4.0]
Output: out = [1.0, 3.0, 4.0, 0.0, 0.0, 0.0]

Constraints
-----------

- 1 ≤ `N` ≤ 100,000,000
- −1000.0 ≤ `A[i]` ≤ 1000.0
- `out` is pre-allocated with `N` elements, initialised to `0.0`
- Performance is measured with `N` = 50,000,000

Run `python3 puzzles/leetgpu/72_stream_compaction.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Stream Compaction"
    atol = 0.0
    rtol = 0.0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, A: torch.Tensor, N: int, out: torch.Tensor):
        assert A.shape == (N,), f"Expected A.shape=({N},), got {A.shape}"
        assert out.shape == (N,), f"Expected out.shape=({N},), got {out.shape}"
        assert A.dtype == torch.float32
        assert out.dtype == torch.float32

        mask = A > 0
        selected = A[mask]
        k = selected.numel()
        out[:k].copy_(selected)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "N": (ctypes.c_int, "in"),
            "out": (ctypes.POINTER(ctypes.c_float), "out"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        A = torch.tensor([1.0, -2.0, 3.0, 0.0, -1.0, 4.0], device=self.device, dtype=dtype)
        N = 6
        out = torch.zeros(N, device=self.device, dtype=dtype)
        return {"A": A, "N": N, "out": out}

    def _make_test(self, N: int, lo: float = -2.0, hi: float = 2.0) -> Dict[str, Any]:
        dtype = torch.float32
        A = torch.empty(N, device=self.device, dtype=dtype).uniform_(lo, hi)
        out = torch.zeros(N, device=self.device, dtype=dtype)
        return {"A": A, "N": N, "out": out}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # Edge cases — tiny sizes
        # N=1, zero (not positive, nothing selected)
        tests.append(
            {
                "A": torch.tensor([0.0], device=self.device, dtype=dtype),
                "N": 1,
                "out": torch.zeros(1, device=self.device, dtype=dtype),
            }
        )
        # N=1, positive (all selected)
        tests.append(
            {
                "A": torch.tensor([5.0], device=self.device, dtype=dtype),
                "N": 1,
                "out": torch.zeros(1, device=self.device, dtype=dtype),
            }
        )
        # N=4, mixed with exact zeros and negatives
        tests.append(
            {
                "A": torch.tensor([-1.0, 2.0, 0.0, 4.0], device=self.device, dtype=dtype),
                "N": 4,
                "out": torch.zeros(4, device=self.device, dtype=dtype),
            }
        )

        # Power-of-2 sizes
        # All positive — every element passes the predicate
        A_all_pos = torch.rand(16, device=self.device, dtype=dtype) + 0.1
        tests.append(
            {"A": A_all_pos, "N": 16, "out": torch.zeros(16, device=self.device, dtype=dtype)}
        )

        # All negative — no element passes the predicate
        A_all_neg = -(torch.rand(32, device=self.device, dtype=dtype) + 0.1)
        tests.append(
            {"A": A_all_neg, "N": 32, "out": torch.zeros(32, device=self.device, dtype=dtype)}
        )

        # Mixed, wide range
        tests.append(self._make_test(256, lo=-5.0, hi=5.0))
        tests.append(self._make_test(1024, lo=-10.0, hi=10.0))

        # Non-power-of-2
        tests.append(self._make_test(100, lo=-3.0, hi=3.0))
        tests.append(self._make_test(255, lo=-1.0, hi=1.0))

        # Realistic size
        tests.append(self._make_test(10000, lo=-100.0, hi=100.0))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 50_000_000
        A = torch.empty(N, device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        out = torch.zeros(N, device=self.device, dtype=dtype)
        return {"A": A, "N": N, "out": out}


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_stream_compaction(...):
#     ...


def solve(
    A: torch.Tensor,
    N: int,
    out: torch.Tensor,
):
    # TODO: launch your TileLang kernel and write the result into `out` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
