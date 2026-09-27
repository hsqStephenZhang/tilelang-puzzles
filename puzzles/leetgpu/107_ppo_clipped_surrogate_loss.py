r"""
LeetGPU 107: PPO Clipped Surrogate Loss
=======================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Given precomputed advantages and log probabilities from the current and previous policies, compute
the scalar PPO clipped-surrogate loss. For each response token, PPO compares the current policy
with the policy that generated the data, limits how far that ratio can move from one, and averages
the resulting surrogate objective across the batch.

Implementation Requirements
---------------------------

- Implement `solve(advantages, log_pi, log_pi_old, output, clip_eps, B, S)` with the signature unchanged
- Write the single scalar loss into `output[0]`
- Use only native features of the selected framework; external libraries are not permitted
- All input tensors have shape `[B, S]` and the output has shape `[1]`
- All `S` positions are valid response tokens; padding and token masks are outside the scope of this challenge
- `advantages` and `log_pi_old` are fixed precomputed inputs; do not differentiate through them

Loss Calculation
----------------

At position `[b, s]`, `log_pi` and `log_pi_old` are the log
probabilities assigned to the same response token by the current and previous policies. Let
`A` denote `advantages`, and let `epsilon` denote
`clip_eps`.

First, compute the probability ratio between the current and previous policies:

\[
r_{b,s} = \exp\!\left(\log\pi_{b,s} - \log\pi^{\mathrm{old}}_{b,s}\right)
\]

Clamp that ratio to PPO's trust region:

\[
\widehat{r}_{b,s} = \operatorname{clip}\!\left(r_{b,s}, 1 - \varepsilon, 1 + \varepsilon\right)
\]

The per-token surrogate takes the less optimistic of the unclipped and clipped values. This
`min` is intentional: it applies correctly to both positive and negative advantages.

\[
L_{b,s}^{\mathrm{CLIP}}
= \min\!\left(r_{b,s} A_{b,s},\; \widehat{r}_{b,s} A_{b,s}\right)
\]

PPO maximizes this surrogate objective. Because this challenge returns a loss for minimization,
negate its mean over every batch and sequence position:

\[
\text{output}[0]
= -\frac{1}{B S}\sum_{b=0}^{B-1}\sum_{s=0}^{S-1} L_{b,s}^{\mathrm{CLIP}}
\]

Examples
--------

For `B` = 1, `S` = 4, and `clip_eps` = 0.2:

\[
\begin{aligned}
\text{advantages}
&= \begin{bmatrix} 1.0 & -2.0 & 3.0 & -4.0 \end{bmatrix} \\[6pt]
\text{log_pi}
&= \begin{bmatrix} \log(1.3) & \log(0.7) & \log(1.1) & \log(0.8) \end{bmatrix} \\[6pt]
\text{log_pi_old}
&= \begin{bmatrix} 0.0 & 0.0 & 0.0 & 0.0 \end{bmatrix}
\end{aligned}
\]

The intermediate values are:

\[
\begin{aligned}
r
&= \begin{bmatrix} 1.3 & 0.7 & 1.1 & 0.8 \end{bmatrix} \\[6pt]
\widehat{r}
&= \begin{bmatrix} 1.2 & 0.8 & 1.1 & 0.8 \end{bmatrix} \\[6pt]
L^{\mathrm{CLIP}}
&= \begin{bmatrix} 1.2 & -1.6 & 3.3 & -3.2 \end{bmatrix}
\end{aligned}
\]

\[
\text{output}[0] = -\operatorname{mean}\!\left(L^{\mathrm{CLIP}}\right) = 0.075
\]

Constraints
-----------

- 1 ≤ `B` ≤ 256
- 1 ≤ `S` ≤ 16,384
- All tensors contain 32-bit floating point values
- `0 ≤ clip_eps < 1`
- `log_pi - log_pi_old` lies in [-16, 16], so every exponential is finite in float32
- Performance is measured with `B` = 256, `S` = 16,384

Reference
---------

Proximal Policy Optimization Algorithms, Schulman et al. (2017)

Run `python3 puzzles/leetgpu/107_ppo_clipped_surrogate_loss.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
import math
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, OutTensor, RandnTensor, run_challenge


class Challenge(ChallengeBase):
    name = "PPO Clipped Surrogate Loss"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        advantages: torch.Tensor,
        log_pi: torch.Tensor,
        log_pi_old: torch.Tensor,
        output: torch.Tensor,
        clip_eps: float,
        B: int,
        S: int,
    ):
        assert advantages.shape == (B, S)
        assert log_pi.shape == (B, S)
        assert log_pi_old.shape == (B, S)
        assert output.shape == (1,)
        assert advantages.dtype == log_pi.dtype == log_pi_old.dtype == output.dtype == torch.float32

        ratio = torch.exp(log_pi - log_pi_old)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
        output[0] = -torch.mean(surrogate)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "advantages": (ctypes.POINTER(ctypes.c_float), "in"),
            "log_pi": (ctypes.POINTER(ctypes.c_float), "in"),
            "log_pi_old": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "clip_eps": (ctypes.c_float, "in"),
            "B": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self,
        B,
        S,
        advantages=None,
        log_pi=None,
        log_pi_old=None,
        clip_eps=0.2,
    ):
        dtype = torch.float32
        device = self.device

        def tensor_or_random(values):
            if values is None:
                return torch.randn(B, S, device=device, dtype=dtype)
            return torch.tensor(values, device=device, dtype=dtype)

        return {
            "advantages": tensor_or_random(advantages),
            "log_pi": tensor_or_random(log_pi),
            "log_pi_old": tensor_or_random(log_pi_old),
            "output": torch.empty(1, device=device, dtype=dtype),
            "clip_eps": clip_eps,
            "B": B,
            "S": S,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        return self._make_test_case(
            1,
            4,
            advantages=[[1.0, -2.0, 3.0, -4.0]],
            log_pi=[[0.262364, -0.356675, 0.0953102, -0.223144]],
            log_pi_old=[[0.0, 0.0, 0.0, 0.0]],
            clip_eps=0.2,
        )

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Hand-computed clipping example with positive and negative advantages.
        tests.append(self.generate_example_test())

        # Single element with no policy change.
        tests.append(
            self._make_test_case(
                1,
                1,
                advantages=[[2.5]],
                log_pi=[[0.0]],
                log_pi_old=[[0.0]],
            )
        )

        # Zero objective inputs.
        tests.append(
            self._make_test_case(
                2,
                4,
                advantages=[[0.0] * 4, [0.0] * 4],
                log_pi=[[0.0] * 4, [0.0] * 4],
                log_pi_old=[[0.0] * 4, [0.0] * 4],
            )
        )

        # Ratios exactly at both clipping boundaries.
        tests.append(
            self._make_test_case(
                1,
                4,
                advantages=[[1.0, -1.0, 2.0, -2.0]],
                log_pi=[[math.log(1.2), math.log(0.8), 0.0, 0.0]],
                log_pi_old=[[0.0] * 4],
            )
        )

        # Zero clipping range: every ratio is clamped to one.
        tests.append(
            self._make_test_case(
                1,
                4,
                advantages=[[1.0, -1.0, 3.0, -2.0]],
                log_pi=[[math.log(1.5), math.log(0.5), math.log(2.0), math.log(0.25)]],
                log_pi_old=[[0.0] * 4],
                clip_eps=0.0,
            )
        )

        # Clipping is inactive for a positive advantage below the range and a negative advantage
        # above it.
        tests.append(
            self._make_test_case(
                1,
                2,
                advantages=[[2.0, -3.0]],
                log_pi=[[math.log(0.5), math.log(1.5)]],
                log_pi_old=[[0.0, 0.0]],
                clip_eps=0.2,
            )
        )

        # Nonzero values across batches verify reduction over both B and S.
        tests.append(
            self._make_test_case(
                2,
                2,
                advantages=[[1.0, -1.0], [2.0, -2.0]],
                log_pi=[
                    [math.log(1.5), math.log(0.5)],
                    [math.log(0.5), math.log(1.5)],
                ],
                log_pi_old=[[0.0, 0.0], [0.0, 0.0]],
                clip_eps=0.2,
            )
        )

        # Power-of-two shape with random mixed-sign values.
        tests.append(self._make_test_case(4, 16))
        tests.append(self._make_test_case(8, 64, clip_eps=0.1))

        # Non-power-of-two shapes.
        tests.append(self._make_test_case(3, 30))
        tests.append(self._make_test_case(5, 100, clip_eps=0.3))

        # Realistic PPO rollout shape.
        tests.append(self._make_test_case(16, 512))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        B, S = 256, 16384
        return {
            "advantages": RandnTensor((B, S)),
            "log_pi": RandnTensor((B, S)),
            "log_pi_old": RandnTensor((B, S)),
            "output": OutTensor((1,)),
            "clip_eps": 0.2,
            "B": B,
            "S": S,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_ppo_clipped_surrogate_loss(...):
#     ...


def solve(
    advantages: torch.Tensor,
    log_pi: torch.Tensor,
    log_pi_old: torch.Tensor,
    output: torch.Tensor,
    clip_eps: float,
    B: int,
    S: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
