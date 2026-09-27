r"""
LeetGPU 64: Weight Dequantization
=================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a GPU program that "dequantizes" a weight matrix on the GPU. You are given an input matrix `X` of shape `[M, N]` containing quantized values and a scale matrix `S` of shape `[ceil(M/T), ceil(N/T)]`, where `T` is the tile size.

For each element \(X_{i,j}\), the corresponding scale factor is \(S_{row, col}\) where \(row = \lfloor i / T \rfloor\) and \(col = \lfloor j / T \rfloor\).
The output \(Y_{i,j}\) should be computed as:
\[
Y_{i,j} = X_{i,j} \times S_{row, col}
\]

Implementation Requirements
---------------------------

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in the output buffer `Y`

Example 1:
----------

Input:
M = 4, N = 4, TILE_SIZE = 2
X = [
[10, 10,  5,  5],
[10, 10,  5,  5],
[ 2,  2,  8,  8],
[ 2,  2,  8,  8]
]
S = [
[0.5, 2.0],
[4.0, 0.25]
]

Output:
Y = [
[ 5.0,  5.0, 10.0, 10.0],
[ 5.0,  5.0, 10.0, 10.0],
[ 8.0,  8.0,  2.0,  2.0],
[ 8.0,  8.0,  2.0,  2.0]
]
Explanation:
Tile (0,0) of X is multiplied by S[0,0] (0.5).
Tile (0,1) of X is multiplied by S[0,1] (2.0).
Tile (1,0) is multiplied by S[1,0] (4.0).
Tile (1,1) is multiplied by S[1,1] (0.25).

Constraints
-----------

- 1 ≤ `M`, `N` ≤ 8192
- `TILE_SIZE` ∈ {16, 32, 64, 128}
- Performance is measured with `M` = 8,192, `N` = 8,192, `TILE_SIZE` = 128

Run `python3 puzzles/leetgpu/64_weight_dequantization.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Weight Dequantization"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self, X: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, M: int, N: int, TILE_SIZE: int
    ):
        s_rows = (M + TILE_SIZE - 1) // TILE_SIZE
        s_cols = (N + TILE_SIZE - 1) // TILE_SIZE
        assert X.shape == (M, N)
        assert S.shape == (s_rows, s_cols)
        assert Y.shape == (M, N)
        assert X.dtype == torch.float32
        assert S.dtype == torch.float32
        assert Y.dtype == torch.float32

        S_expanded = S.repeat_interleave(TILE_SIZE, dim=0)[:M, :]
        S_expanded = S_expanded.repeat_interleave(TILE_SIZE, dim=1)[:, :N]

        Y.copy_(X * S_expanded)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "X": (ctypes.POINTER(ctypes.c_float), "in"),
            "S": (ctypes.POINTER(ctypes.c_float), "in"),
            "Y": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "TILE_SIZE": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        M, N = 256, 256
        TILE_SIZE = 128
        X = torch.randn(M, N, device=self.device, dtype=torch.float32)
        # S shape
        s_rows = (M + TILE_SIZE - 1) // TILE_SIZE
        s_cols = (N + TILE_SIZE - 1) // TILE_SIZE
        S = torch.randn(s_rows, s_cols, device=self.device, dtype=torch.float32)
        Y = torch.empty_like(X)

        return {
            "X": X,
            "S": S,
            "Y": Y,
            "M": M,
            "N": N,
            "TILE_SIZE": TILE_SIZE,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        test_configs = [
            # Edge cases - small sizes
            (1, 1, 16),
            (2, 3, 16),
            (4, 4, 16),
            # Power-of-2 sizes
            (64, 64, 32),
            (128, 128, 64),
            (256, 256, 128),
            (512, 512, 128),
            # Non-power-of-2 sizes (padding needed)
            (30, 50, 16),
            (100, 100, 32),
            (130, 200, 128),
            (255, 255, 64),
            # Realistic sizes
            (1024, 1024, 128),
            (2048, 4096, 128),
        ]

        for M, N, TILE_SIZE in test_configs:
            s_rows = (M + TILE_SIZE - 1) // TILE_SIZE
            s_cols = (N + TILE_SIZE - 1) // TILE_SIZE
            tests.append(
                {
                    "X": torch.randn(M, N, device=self.device, dtype=torch.float32),
                    "S": torch.randn(s_rows, s_cols, device=self.device, dtype=torch.float32),
                    "Y": torch.zeros(M, N, device=self.device, dtype=torch.float32),
                    "M": M,
                    "N": N,
                    "TILE_SIZE": TILE_SIZE,
                }
            )

        # Zero input
        M, N, TILE_SIZE = 64, 64, 32
        s_rows = (M + TILE_SIZE - 1) // TILE_SIZE
        s_cols = (N + TILE_SIZE - 1) // TILE_SIZE
        tests.append(
            {
                "X": torch.zeros(M, N, device=self.device, dtype=torch.float32),
                "S": torch.randn(s_rows, s_cols, device=self.device, dtype=torch.float32),
                "Y": torch.zeros(M, N, device=self.device, dtype=torch.float32),
                "M": M,
                "N": N,
                "TILE_SIZE": TILE_SIZE,
            }
        )

        # Negative values
        M, N, TILE_SIZE = 128, 128, 64
        s_rows = (M + TILE_SIZE - 1) // TILE_SIZE
        s_cols = (N + TILE_SIZE - 1) // TILE_SIZE
        tests.append(
            {
                "X": torch.randn(M, N, device=self.device, dtype=torch.float32).sub_(0.5),
                "S": torch.randn(s_rows, s_cols, device=self.device, dtype=torch.float32).sub_(0.5),
                "Y": torch.zeros(M, N, device=self.device, dtype=torch.float32),
                "M": M,
                "N": N,
                "TILE_SIZE": TILE_SIZE,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        M, N = 8192, 8192
        TILE_SIZE = 128
        X = torch.randn(M, N, device=self.device, dtype=torch.float32)
        s_rows = (M + TILE_SIZE - 1) // TILE_SIZE
        s_cols = (N + TILE_SIZE - 1) // TILE_SIZE
        S = torch.randn(s_rows, s_cols, device=self.device, dtype=torch.float32)
        Y = torch.empty_like(X)

        return {
            "X": X,
            "S": S,
            "Y": Y,
            "M": M,
            "N": N,
            "TILE_SIZE": TILE_SIZE,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_weight_dequantization(...):
#     ...


def solve(
    X: torch.Tensor,
    S: torch.Tensor,
    Y: torch.Tensor,
    M: int,
    N: int,
    TILE_SIZE: int,
):
    # TODO: launch your TileLang kernel and write the result into `Y` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
