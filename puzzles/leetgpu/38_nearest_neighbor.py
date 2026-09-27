r"""
LeetGPU 38: Nearest Neighbor
============================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a GPU program that, for `N` three-dimensional points stored on the device, fills `indices[i]` with the index `j ≠ i` of the point closest to `points[i]`. Comparing squared Euclidean distance is sufficient—you do not need to compute square-roots.

Implementation Requirements
---------------------------

- The `solve` function signature must remain unchanged
- External libraries are not permitted
- The final result must be stored in the `indices` array

Example 1:
----------

Input:  points  = [(0,0,0), (1,0,0), (5,5,5)]
indices = [-1, -1, -1]
N       = 3
Output: indices = [1, 0, 1]   # 0⇆1 are nearest, 2 is closest to 1

Constraints
-----------

- 1 ≤ `N` ≤ 100,000
- Coordinates are 32-bit floats in the range [-1000, 1000]
- Performance is measured with `N` = 10,000

Run `python3 puzzles/leetgpu/38_nearest_neighbor.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, OutTensor, RandTensor, run_challenge


class Challenge(ChallengeBase):
    name = "Nearest Neighbor"
    atol = 0
    rtol = 0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, points: torch.Tensor, indices: torch.Tensor, N: int):
        """
        Reference implementation that finds the nearest neighbor for each point.
        For N three-dimensional points, fills indices[i] with the index j≠i
        of the point closest to points[i].
        """
        assert points.dtype == torch.float32
        assert indices.dtype == torch.int32
        assert points.shape == (N * 3,)  # N points, each with 3 coordinates
        assert indices.shape == (N,)
        assert N >= 1

        # Reshape points to (N, 3) for easier processing
        pts = points.view(N, 3)

        # pts shape: (N, 3)
        # Expand to (N, 1, 3) and (1, N, 3) for broadcasting
        pts_expand1 = pts.unsqueeze(1)  # (N, 1, 3)
        pts_expand2 = pts.unsqueeze(0)  # (1, N, 3)

        # Compute all pairwise squared distances: (N, N)
        diff = pts_expand1 - pts_expand2  # (N, N, 3)
        dist_sq = torch.sum(diff * diff, dim=2)  # (N, N)

        # Mask diagonal (distance to self) with large value
        mask = torch.eye(N, device=points.device, dtype=torch.bool)
        dist_sq[mask] = float("inf")

        # Find nearest neighbor indices
        indices.copy_(torch.argmin(dist_sq, dim=1).int())

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "points": (ctypes.POINTER(ctypes.c_float), "in"),
            "indices": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype_float = torch.float32
        dtype_int = torch.int32
        N = 3

        # Example: points = [(0,0,0), (1,0,0), (5,5,5)]
        points_data = torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 5.0, 5.0],  # point 0  # point 1  # point 2
            device=self.device,
            dtype=dtype_float,
        )
        indices_data = torch.full((N,), -1, device=self.device, dtype=dtype_int)

        return {
            "points": points_data,
            "indices": indices_data,
            "N": N,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype_float = torch.float32
        dtype_int = torch.int32
        test_cases = []

        # Test case 1: Basic example from problem description
        test_cases.append(
            {
                "points": torch.tensor(
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 5.0, 5.0],  # point 0  # point 1  # point 2
                    device=self.device,
                    dtype=dtype_float,
                ),
                "indices": torch.full((3,), -1, device=self.device, dtype=dtype_int),
                "N": 3,
            }
        )

        # Test case 2: Two points only
        test_cases.append(
            {
                "points": torch.tensor(
                    [0.0, 0.0, 0.0, 3.0, 4.0, 0.0],  # point 0  # point 1
                    device=self.device,
                    dtype=dtype_float,
                ),
                "indices": torch.full((2,), -1, device=self.device, dtype=dtype_int),
                "N": 2,
            }
        )

        # Test case 3: Four points in a square
        test_cases.append(
            {
                "points": torch.tensor(
                    [
                        0.0,
                        0.0,
                        0.0,  # point 0
                        1.0,
                        0.0,
                        0.0,  # point 1
                        0.0,
                        1.0,
                        0.0,  # point 2
                        1.0,
                        1.0,
                        0.0,
                    ],  # point 3
                    device=self.device,
                    dtype=dtype_float,
                ),
                "indices": torch.full((4,), -1, device=self.device, dtype=dtype_int),
                "N": 4,
            }
        )

        # Test case 4: Points with negative coordinates
        test_cases.append(
            {
                "points": torch.tensor(
                    [
                        -1.0,
                        -1.0,
                        -1.0,  # point 0
                        1.0,
                        1.0,
                        1.0,  # point 1
                        0.0,
                        0.0,
                        0.0,  # point 2
                        2.0,
                        2.0,
                        2.0,
                    ],  # point 3
                    device=self.device,
                    dtype=dtype_float,
                ),
                "indices": torch.full((4,), -1, device=self.device, dtype=dtype_int),
                "N": 4,
            }
        )

        # Test case 5: Points with clear unique nearest neighbors
        test_cases.append(
            {
                "points": torch.tensor(
                    [
                        0.0,
                        0.0,
                        0.0,  # point 0
                        10.0,
                        0.0,
                        0.0,  # point 1
                        1.0,
                        0.0,
                        0.0,  # point 2 (closest to 0)
                        11.0,
                        0.0,
                        0.0,  # point 3 (closest to 1)
                        5.0,
                        0.0,
                        0.0,
                    ],  # point 4
                    device=self.device,
                    dtype=dtype_float,
                ),
                "indices": torch.full((5,), -1, device=self.device, dtype=dtype_int),
                "N": 5,
            }
        )

        # Test case 6: Medium random test with fixed seed for reproducibility
        torch.manual_seed(42)
        test_cases.append(
            {
                "points": torch.empty((100, 3), device=self.device, dtype=dtype_float)
                .uniform_(-100.0, 100.0)
                .flatten(),
                "indices": torch.full((100,), -1, device=self.device, dtype=dtype_int),
                "N": 100,
            }
        )

        # Test case 7: Larger test with fixed seed
        torch.manual_seed(123)
        test_cases.append(
            {
                "points": torch.empty((250, 3), device=self.device, dtype=dtype_float)
                .uniform_(-1000.0, 1000.0)
                .flatten(),
                "indices": torch.full((250,), -1, device=self.device, dtype=dtype_int),
                "N": 250,
            }
        )

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        N = 10000
        return {
            "points": RandTensor((N * 3,), -1000.0, 1000.0),
            "indices": OutTensor((N,), dtype="int32"),
            "N": N,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_nearest_neighbor(...):
#     ...


def solve(
    points: torch.Tensor,
    indices: torch.Tensor,
    N: int,
):
    # TODO: launch your TileLang kernel and write the result into `indices` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
