r"""
LeetGPU 27: Mean Squared Error
==============================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a GPU program to calculate the Mean Squared Error (MSE) between
predicted values and target values. Given two arrays of equal length,
`predictions` and `targets`, compute: \[ \text{MSE} =
\frac{1}{N} \sum_{i=1}^{N} (predictions_i - targets_i)^2 \] where N is the
number of elements in each array.

Implementation Requirements
---------------------------

- External libraries are not permitted.
- The `solve` function signature must remain unchanged.
- The final result must be stored in the `mse` variable.

Example 1:
----------

Input:  predictions = [1.0, 2.0, 3.0, 4.0]
targets = [1.5, 2.5, 3.5, 4.5]
Output: mse = 0.25

Example 2:
----------

Input:  predictions = [10.0, 20.0, 30.0]
targets = [12.0, 18.0, 33.0]
Output: mse = 5.67

Constraints
-----------

- 1 ≤ `N` ≤ 100,000,000

-
-1000.0 ≤ `predictions[i]`, `targets[i]` ≤
1000.0

- Performance is measured with `N` = 50,000,000

Run `python3 puzzles/leetgpu/27_mean_squared_error.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Mean Squared Error"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self, predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int
    ):
        # predictions, targets, mse are tensors on the GPU
        squared_diffs = torch.square(predictions - targets)
        mean_squared_error = torch.mean(squared_diffs)
        mse[0] = mean_squared_error

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "predictions": (ctypes.POINTER(ctypes.c_float), "in"),
            "targets": (ctypes.POINTER(ctypes.c_float), "in"),
            "mse": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device, dtype=dtype)
        targets = torch.tensor([1.5, 2.5, 3.5, 4.5], device=self.device, dtype=dtype)
        mse = torch.empty(1, device=self.device, dtype=dtype)
        N = 4
        return {
            "predictions": predictions,
            "targets": targets,
            "mse": mse,
            "N": N,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # Test 1: basic_example
        tests.append(
            {
                "predictions": torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device, dtype=dtype),
                "targets": torch.tensor([1.5, 2.5, 3.5, 4.5], device=self.device, dtype=dtype),
                "mse": torch.empty(1, device=self.device, dtype=dtype),
                "N": 4,
            }
        )
        # Test 2: second_example
        tests.append(
            {
                "predictions": torch.tensor([10.0, 20.0, 30.0], device=self.device, dtype=dtype),
                "targets": torch.tensor([12.0, 18.0, 33.0], device=self.device, dtype=dtype),
                "mse": torch.empty(1, device=self.device, dtype=dtype),
                "N": 3,
            }
        )
        # Test 3: zero_error
        tests.append(
            {
                "predictions": torch.tensor(
                    [1.5, 2.5, 3.5, 4.5, 5.5], device=self.device, dtype=dtype
                ),
                "targets": torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5], device=self.device, dtype=dtype),
                "mse": torch.empty(1, device=self.device, dtype=dtype),
                "N": 5,
            }
        )
        # Test 4: negative_values
        tests.append(
            {
                "predictions": torch.tensor(
                    [-2.5, -1.0, 0.0, 1.5], device=self.device, dtype=dtype
                ),
                "targets": torch.tensor([-1.5, -2.0, 0.5, 2.0], device=self.device, dtype=dtype),
                "mse": torch.empty(1, device=self.device, dtype=dtype),
                "N": 4,
            }
        )
        # Test 5: large_difference
        tests.append(
            {
                "predictions": torch.tensor([100.0, 200.0, 300.0], device=self.device, dtype=dtype),
                "targets": torch.tensor([150.0, 250.0, 350.0], device=self.device, dtype=dtype),
                "mse": torch.empty(1, device=self.device, dtype=dtype),
                "N": 3,
            }
        )
        # Test 6: medium_size
        N = 1024
        predictions = torch.empty(N, device=self.device, dtype=dtype).uniform_(-10.0, 10.0)
        targets = torch.empty(N, device=self.device, dtype=dtype).uniform_(-10.0, 10.0)
        mse = torch.empty(1, device=self.device, dtype=dtype)
        tests.append({"predictions": predictions, "targets": targets, "mse": mse, "N": N})
        # Test 7: large_size
        N = 10000
        predictions = torch.empty(N, device=self.device, dtype=dtype).uniform_(-100.0, 100.0)
        targets = torch.empty(N, device=self.device, dtype=dtype).uniform_(-100.0, 100.0)
        mse = torch.empty(1, device=self.device, dtype=dtype)
        tests.append({"predictions": predictions, "targets": targets, "mse": mse, "N": N})
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 50000000
        predictions = torch.empty(N, device=self.device, dtype=dtype).uniform_(-1000.0, 1000.0)
        targets = torch.empty(N, device=self.device, dtype=dtype).uniform_(-1000.0, 1000.0)
        mse = torch.empty(1, device=self.device, dtype=dtype)
        return {
            "predictions": predictions,
            "targets": targets,
            "mse": mse,
            "N": N,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_mean_squared_error(...):
#     ...


def solve(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mse: torch.Tensor,
    N: int,
):
    # TODO: launch your TileLang kernel and write the result into `mse` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
