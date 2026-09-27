r"""
LeetGPU 50: RMS Normalization
=============================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement RMS Normalization forward pass for 1D input vectors. Given an input tensor of shape [N] where N is the number of elements, compute the normalized output using a scalar scale (`gamma`) and shift (`beta`) parameter.

RMS Normalization computes:
\[
\begin{align}
\text{rms} &= \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon} \\
\hat{x}_i &= \frac{x_i}{\text{rms}} \\
y_i &= \gamma \hat{x}_i + \beta
\end{align}
\]

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` tensor

Example 1:
----------

Input:  input = [1.0, 2.0, 3.0, 4.0]  (N=4)
gamma = 1.0
beta = 0.0
eps = 1e-5
Output: output = [0.36514813, 0.73029625, 1.0954444, 1.4605925 ]

Example 2:
----------

Input:  input = [1.0, 2.0, 3.0]  (N=3)
gamma = 1.0
beta = 0.0
eps = 1e-5
Output: output = [0.46290955, 0.9258191, 1.3887286]

Constraints
-----------

- 1 ≤ `N` ≤ 100,000
- `eps` = 1e-5
- -100.0 ≤ input values ≤ 100.0
- 0.1 ≤ gamma ≤ 10.0
- -10.0 ≤ beta ≤ 10.0
- Performance is measured with `N` = 100,000

Run `python3 puzzles/leetgpu/50_rms_normalization.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "RMS Normalization"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        input: torch.Tensor,
        gamma: float,
        beta: float,
        output: torch.Tensor,
        N: int,
        eps: float,
    ):
        assert input.shape == output.shape == (N,)
        assert input.dtype == output.dtype
        assert input.device == output.device

        # RMSNorm: compute root mean square (without mean-centering)
        rms = torch.sqrt(torch.mean(input**2) + eps)  # shape: scalar

        # Normalize
        normalized = input / rms  # shape: [N]

        # Scale and shift
        output.copy_(gamma * normalized + beta)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "gamma": (ctypes.c_float, "in"),
            "beta": (ctypes.c_float, "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 4
        input = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device, dtype=dtype)
        gamma = 1.0
        beta = 0.0
        output = torch.empty(N, device=self.device, dtype=dtype)
        eps = 1e-5
        return {
            "input": input,
            "gamma": gamma,
            "beta": beta,
            "output": output,
            "N": N,
            "eps": eps,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_small
        N = 3
        tests.append(
            {
                "input": torch.tensor([1.0, 2.0, 3.0], device=self.device, dtype=dtype),
                "gamma": 1.0,
                "beta": 0.0,
                "output": torch.empty(N, device=self.device, dtype=dtype),
                "N": N,
                "eps": 1e-5,
            }
        )

        # single_feature
        N = 1
        tests.append(
            {
                "input": torch.tensor([5.0], device=self.device, dtype=dtype),
                "gamma": 2.0,
                "beta": -1.0,
                "output": torch.empty(N, device=self.device, dtype=dtype),
                "N": N,
                "eps": 1e-5,
            }
        )

        # all zeros
        N = 4
        tests.append(
            {
                "input": torch.zeros(N, device=self.device, dtype=dtype),
                "gamma": 1.0,
                "beta": 0.0,
                "output": torch.empty(N, device=self.device, dtype=dtype),
                "N": N,
                "eps": 1e-5,
            }
        )

        # negative numbers
        N = 5
        tests.append(
            {
                "input": torch.tensor(
                    [-1.0, -2.0, -3.0, -4.0, -5.0], device=self.device, dtype=dtype
                ),
                "gamma": 1.0,
                "beta": 0.0,
                "output": torch.empty(N, device=self.device, dtype=dtype),
                "N": N,
                "eps": 1e-5,
            }
        )

        # different gamma/beta
        N = 3
        tests.append(
            {
                "input": torch.tensor([0.0, 1.0, 2.0], device=self.device, dtype=dtype),
                "gamma": 0.5,
                "beta": -1.0,
                "output": torch.empty(N, device=self.device, dtype=dtype),
                "N": N,
                "eps": 1e-5,
            }
        )

        # large values
        N = 8
        tests.append(
            {
                "input": torch.empty(N, device=self.device, dtype=dtype).uniform_(-100.0, 100.0),
                "gamma": 1.5,
                "beta": 0.0,
                "output": torch.empty(N, device=self.device, dtype=dtype),
                "N": N,
                "eps": 1e-5,
            }
        )

        # large N
        N = 2000
        tests.append(
            {
                "input": torch.empty(N, device=self.device, dtype=dtype).uniform_(-100.0, 100.0),
                "gamma": 1.3,
                "beta": 0.0,
                "output": torch.empty(N, device=self.device, dtype=dtype),
                "N": N,
                "eps": 1e-5,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 100000
        return {
            "input": torch.empty(N, device=self.device, dtype=dtype).uniform_(-10.0, 10.0),
            "gamma": 1.5,
            "beta": 0.0,
            "output": torch.empty(N, device=self.device, dtype=dtype),
            "N": N,
            "eps": 1e-5,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_rms_normalization(...):
#     ...


def solve(
    input: torch.Tensor,
    gamma: float,
    beta: float,
    output: torch.Tensor,
    N: int,
    eps: float,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
