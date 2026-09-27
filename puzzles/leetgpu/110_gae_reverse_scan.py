r"""
LeetGPU 110: Parallel Reverse Scan (GAE)
========================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Given reward and value sequences with shape `[B, S]`, compute the Generalized Advantage
Estimate (GAE) for every batch element and timestep. GAE combines one-step temporal-difference
errors with a reverse discounted accumulation, creating a dependency that runs from the end of
each sequence toward the beginning.

(diagram omitted, see challenge.html in leetgpu-challenges)

How GAE Is Computed
-------------------

The calculation has two stages. First, compute the one-step temporal-difference error at every
position. The value after the final position is defined as zero:

\[
\widetilde{V}_{b,t+1} =
\begin{cases}
V_{b,t+1}, & t < S - 1 \\
0, & t = S - 1
\end{cases}
\qquad
\delta_{b,t} = r_{b,t} + \gamma \widetilde{V}_{b,t+1} - V_{b,t}
\]

Next, let `c = gamma × lambda`. The advantage at position `t` is the
current TD error plus the decayed advantage from the next position:

\[
A_{b,S} = 0, \qquad A_{b,S-1} = \delta_{b,S-1}, \qquad
A_{b,t} = \delta_{b,t} + c A_{b,t+1} \quad \text{for } t = S-2, S-3, \ldots, 0
\]

Therefore, implementations can compute all `delta` values in parallel, then perform a
reverse scan over the sequence using the recurrence above. For long sequences, the reverse scan
can be split into blocks: each block computes its local scan, then receives the decayed carry from
the block to its right.

For the example below, `gamma` = 0.9 and `lambda` = 0.5, so
`c` = 0.45:

\[
\delta = \begin{bmatrix} 1.4 & 2.35 & 3.3 & 2.0 \end{bmatrix}
\]
\[
A_3 = 2.0, \quad
A_2 = 3.3 + 0.45(2.0) = 4.2, \quad
A_1 = 2.35 + 0.45(4.2) = 4.24, \quad
A_0 = 1.4 + 0.45(4.24) = 3.308
\]

Implementation Requirements
---------------------------

- Implement `solve(rewards, values, advantages, gamma, lam, B, S)` with the signature unchanged
- Write the result into the preallocated `advantages` tensor
- Use only native features of the selected framework; external libraries are not permitted
- For the final timestep, use a bootstrap value of zero
- Use float32 arithmetic for all tensors
- Every sequence terminates at `S`; padding, per-token terminal flags, and masks are outside the scope of this challenge

Examples
--------

For `B` = 1, `S` = 4, `gamma` = 0.9, and `lambda` = 0.5:

\[
\text{rewards} = \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 \end{bmatrix}, \quad
\text{values} = \begin{bmatrix} 0.5 & 1.0 & 1.5 & 2.0 \end{bmatrix}
\]
\[
\text{advantages} = \begin{bmatrix} 3.308 & 4.240 & 4.200 & 2.000 \end{bmatrix}
\]

The final timestep uses `next_value` = 0, so its temporal-difference error is
`4.0 - 2.0 = 2.0`. Each earlier advantage includes the discounted contribution from later
timesteps.

Constraints
-----------

- 1 ≤ `B` ≤ 256
- 1 ≤ `S` ≤ 65,536
- `rewards`, `values`, and `advantages` have shape `[B, S]`
- All tensors contain 32-bit floating point values
- `gamma` and `lambda` are in the range [0, 1]
- Performance is measured with `B` = 64, `S` = 4,096

Reference
---------

High-Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. (2015)

Run `python3 puzzles/leetgpu/110_gae_reverse_scan.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, OutTensor, RandnTensor, run_challenge


class Challenge(ChallengeBase):
    name = "Parallel Reverse Scan (GAE)"
    atol = 0.001
    rtol = 0.001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor,
        gamma: float,
        lam: float,
        B: int,
        S: int,
    ):
        assert rewards.shape == (B, S)
        assert values.shape == (B, S)
        assert advantages.shape == (B, S)
        assert rewards.dtype == values.dtype == advantages.dtype == torch.float32

        next_values = torch.zeros_like(values)
        if S > 1:
            next_values[:, :-1] = values[:, 1:]
        deltas = rewards + gamma * next_values - values

        last_gae = torch.zeros(B, device=self.device, dtype=rewards.dtype)
        decay = gamma * lam
        for t in reversed(range(S)):
            last_gae = deltas[:, t] + decay * last_gae
            advantages[:, t] = last_gae

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "rewards": (ctypes.POINTER(ctypes.c_float), "in"),
            "values": (ctypes.POINTER(ctypes.c_float), "in"),
            "advantages": (ctypes.POINTER(ctypes.c_float), "out"),
            "gamma": (ctypes.c_float, "in"),
            "lam": (ctypes.c_float, "in"),
            "B": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, B, S, rewards=None, values=None, gamma=0.99, lam=0.95):
        dtype = torch.float32
        device = self.device
        if rewards is None:
            rewards = torch.randn(B, S, device=device, dtype=dtype)
        else:
            rewards = torch.tensor(rewards, device=device, dtype=dtype)
        if values is None:
            values = torch.randn(B, S, device=device, dtype=dtype)
        else:
            values = torch.tensor(values, device=device, dtype=dtype)
        return {
            "rewards": rewards,
            "values": values,
            "advantages": torch.empty(B, S, device=device, dtype=dtype),
            "gamma": gamma,
            "lam": lam,
            "B": B,
            "S": S,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        return self._make_test_case(
            1,
            4,
            rewards=[[1.0, 2.0, 3.0, 4.0]],
            values=[[0.5, 1.0, 1.5, 2.0]],
            gamma=0.9,
            lam=0.5,
        )

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        tests.append(self._make_test_case(1, 1, rewards=[[2.0]], values=[[0.5]]))
        tests.append(self._make_test_case(1, 2, rewards=[[1.0, -1.0]], values=[[0.0, 0.5]]))
        tests.append(
            self._make_test_case(2, 4, rewards=[[0.0] * 4, [0.0] * 4], values=[[0.0] * 4] * 2)
        )

        # gamma=0 removes both the bootstrap value and the reverse-scan carry.
        tests.append(
            self._make_test_case(
                1,
                4,
                rewards=[[1.0, -2.0, 3.0, -4.0]],
                values=[[0.5, 1.0, -1.0, 2.0]],
                gamma=0.0,
                lam=0.95,
            )
        )

        # lambda=0 keeps the one-step TD error but removes the reverse-scan carry.
        tests.append(
            self._make_test_case(
                1,
                4,
                rewards=[[1.0, 2.0, 3.0, 4.0]],
                values=[[0.5, 1.0, 1.5, 2.0]],
                gamma=0.9,
                lam=0.0,
            )
        )

        tests.append(
            self._make_test_case(
                1,
                4,
                rewards=[[-1.0, -2.0, 3.0, -4.0]],
                values=[[1.0, -1.0, 2.0, -2.0]],
            )
        )
        tests.append(self._make_test_case(4, 16))
        tests.append(self._make_test_case(8, 64, gamma=1.0, lam=1.0))
        tests.append(self._make_test_case(2, 30))
        tests.append(self._make_test_case(4, 100, gamma=0.9, lam=0.8))
        # A partial final block catches block-carry indexing bugs in parallel scans.
        tests.append(self._make_test_case(2, 257))
        tests.append(self._make_test_case(16, 1024))
        tests.append(self._make_test_case(64, 4096))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        B, S = 64, 4096
        return {
            "rewards": RandnTensor((B, S)),
            "values": RandnTensor((B, S)),
            "advantages": OutTensor((B, S)),
            "gamma": 0.99,
            "lam": 0.95,
            "B": B,
            "S": S,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_gae_reverse_scan(...):
#     ...


def solve(
    rewards: torch.Tensor,
    values: torch.Tensor,
    advantages: torch.Tensor,
    gamma: float,
    lam: float,
    B: int,
    S: int,
):
    # TODO: launch your TileLang kernel and write the result into `advantages` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
