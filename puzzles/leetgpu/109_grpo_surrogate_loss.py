r"""
LeetGPU 109: GRPO Surrogate Loss
================================

Category: ["leetgpu"]
Difficulty: ["medium"]

For each of `B` prompts, GRPO compares the rewards of `G` sampled responses.
It uses each response's reward relative to the rest of its group as an advantage, rather than
learning a separate critic. That advantage is applied to every one of the response's
`S` tokens in a clipped policy objective with a reference-policy KL regularizer.

At position `[b, g, s]`, `log_pi`, `log_pi_old`, and
`log_ref` all score the same sampled token `y[b,g,s]`, conditioned on
prompt `b` and the preceding response tokens.

(diagram omitted, see challenge.html in leetgpu-challenges)

Implementation Requirements
---------------------------

- Implement `solve(rewards, log_pi, log_pi_old, log_ref, output, clip_eps, beta, B, G, S)` with the signature unchanged
- Write the scalar mean loss into `output[0]`
- Use only native features of the selected framework; external libraries are not permitted
- Rewards have shape `[B, G]`; all log-probability tensors have shape `[B, G, S]`
- Token masks and padding handling are outside the scope of this challenge; all `S` positions participate in the reduction
- Rewards, `log_pi_old`, and `log_ref` are fixed precomputed inputs; do not differentiate through them

Loss Calculation
----------------

For one prompt `b`, normalize the `G` response rewards using the group's
population mean and standard deviation. Let `R` denote `rewards`.

\[
\mu_b = \frac{1}{G}\sum_{g=0}^{G-1} R_{b,g}
\]
\[
\sigma_b = \sqrt{\frac{1}{G}\sum_{g=0}^{G-1}(R_{b,g}-\mu_b)^2}
\]
\[
A_{b,g} = \frac{R_{b,g}-\mu_b}{\sigma_b + 10^{-8}}
\]

This is an outcome-level GRPO advantage. The same advantage is broadcast to every token of
response `g`:

\[
A_{b,g,s} = A_{b,g}
\]

Compare the current policy with the old policy that sampled the response.

\[
r_{b,g,s} = \exp\!\left(\log\pi_{b,g,s} - \log\pi^{\mathrm{old}}_{b,g,s}\right)
\]

Clamp that ratio around one, then take the less optimistic of the clipped and unclipped values.
Let `epsilon` denote `clip_eps`. This `min` applies correctly to
both positive and negative advantages.

\[
\widehat{r}_{b,g,s} = \operatorname{clip}\!\left(r_{b,g,s}, 1-\varepsilon, 1+\varepsilon\right)
\]

\[
L^{\mathrm{clip}}_{b,g,s}
= \min\!\left(r_{b,g,s}A_{b,g,s},\; \widehat{r}_{b,g,s}A_{b,g,s}\right)
\]

GRPO also penalizes divergence from the frozen reference policy. Let `beta` be the KL
coefficient. The non-negative `k3` sample estimator is:

\[
d_{b,g,s} = \log\pi^{\mathrm{ref}}_{b,g,s} - \log\pi_{b,g,s}
\]
\[
K_{b,g,s} = \exp(d_{b,g,s}) - d_{b,g,s} - 1
\]

GRPO maximizes the clipped objective minus this penalty. Because this challenge returns a loss for
minimization, write the negated mean over all prompts, responses, and tokens to `output[0]`:

\[
\text{output}[0]
= -\frac{1}{BGS}\sum_{b=0}^{B-1}\sum_{g=0}^{G-1}\sum_{s=0}^{S-1}
\left(L^{\mathrm{clip}}_{b,g,s} - \beta K_{b,g,s}\right)
\]

Examples
--------

For `B` = 1, `G` = 2, `S` = 2, `clip_eps` = 0.2,
and `beta` = 0.01:

\[
\begin{aligned}
\text{rewards}
&= \begin{bmatrix} 10.0 & 0.0 \end{bmatrix} \\[6pt]
\text{log_pi}
&= \begin{bmatrix} 0.1 & 0.2 \\ -0.5 & -0.4 \end{bmatrix} \\[6pt]
\text{log_pi_old}
&= \begin{bmatrix} 0.0 & 0.0 \\ 0.0 & 0.0 \end{bmatrix} \\[6pt]
\text{log_ref}
&= \begin{bmatrix} 0.0 & 0.0 \\ 0.0 & 0.0 \end{bmatrix}
\end{aligned}
\]
\[
\text{output}[0] \approx -0.17563
\]

Constraints
-----------

- 1 ≤ `B` ≤ 256
- 2 ≤ `G` ≤ 32
- 1 ≤ `S` ≤ 16,384
- All tensors contain 32-bit floating point values
- `0 ≤ clip_eps < 1` and `0 ≤ beta ≤ 1`
- Both `log_pi - log_pi_old` and `log_ref - log_pi` lie in [-16, 16], so every exponential is finite in float32
- Performance is measured with `B` = 64, `G` = 16, `S` = 4,096

Reference
---------

DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO), Shao et al. (2024)

Run `python3 puzzles/leetgpu/109_grpo_surrogate_loss.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, OutTensor, RandnTensor, run_challenge


class Challenge(ChallengeBase):
    name = "GRPO Surrogate Loss"
    atol = 1e-4
    rtol = 1e-4
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        rewards: torch.Tensor,
        log_pi: torch.Tensor,
        log_pi_old: torch.Tensor,
        log_ref: torch.Tensor,
        output: torch.Tensor,
        clip_eps: float,
        beta: float,
        B: int,
        G: int,
        S: int,
    ):
        assert rewards.shape == (B, G)
        assert log_pi.shape == (B, G, S)
        assert log_pi_old.shape == (B, G, S)
        assert log_ref.shape == (B, G, S)
        assert output.shape == (1,)
        assert (
            rewards.dtype
            == log_pi.dtype
            == log_pi_old.dtype
            == log_ref.dtype
            == output.dtype
            == torch.float32
        )

        mean_rewards = rewards.mean(dim=1, keepdim=True)
        std_rewards = rewards.std(dim=1, keepdim=True, unbiased=False)
        advantages = ((rewards - mean_rewards) / (std_rewards + 1e-8)).unsqueeze(-1)

        ratio = torch.exp(log_pi - log_pi_old)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)

        kl_diff = log_ref - log_pi
        kl_penalty = torch.exp(kl_diff) - kl_diff - 1.0
        output[0] = -torch.mean(surrogate - beta * kl_penalty)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "rewards": (ctypes.POINTER(ctypes.c_float), "in"),
            "log_pi": (ctypes.POINTER(ctypes.c_float), "in"),
            "log_pi_old": (ctypes.POINTER(ctypes.c_float), "in"),
            "log_ref": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "clip_eps": (ctypes.c_float, "in"),
            "beta": (ctypes.c_float, "in"),
            "B": (ctypes.c_int, "in"),
            "G": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self,
        B,
        G,
        S,
        rewards=None,
        log_pi=None,
        log_pi_old=None,
        log_ref=None,
        clip_eps=0.2,
        beta=0.01,
    ):
        dtype = torch.float32
        device = self.device

        def tensor_or_random(values, shape):
            if values is None:
                return torch.randn(*shape, device=device, dtype=dtype)
            return torch.tensor(values, device=device, dtype=dtype)

        return {
            "rewards": tensor_or_random(rewards, (B, G)),
            "log_pi": tensor_or_random(log_pi, (B, G, S)),
            "log_pi_old": tensor_or_random(log_pi_old, (B, G, S)),
            "log_ref": tensor_or_random(log_ref, (B, G, S)),
            "output": torch.empty(1, device=device, dtype=dtype),
            "clip_eps": clip_eps,
            "beta": beta,
            "B": B,
            "G": G,
            "S": S,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        return self._make_test_case(
            1,
            2,
            2,
            rewards=[[10.0, 0.0]],
            log_pi=[[[0.1, 0.2], [-0.5, -0.4]]],
            log_pi_old=[[[0.0, 0.0], [0.0, 0.0]]],
            log_ref=[[[0.0, 0.0], [0.0, 0.0]]],
            clip_eps=0.2,
            beta=0.01,
        )

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Hand-computed group normalization and clipping example.
        tests.append(self.generate_example_test())

        # Two groups with two responses each.
        tests.append(self._make_test_case(2, 2, 4))

        # Single response token; the advantage broadcast still has to be correct.
        tests.append(self._make_test_case(1, 2, 1, rewards=[[0.0, 2.0]]))

        # Equal rewards: advantages are zero while KL remains active.
        tests.append(
            self._make_test_case(
                2,
                4,
                8,
                rewards=[[3.0] * 4, [-2.0] * 4],
                log_pi_old=[[[0.0] * 8] * 4] * 2,
            )
        )

        # Negative and mixed rewards.
        tests.append(
            self._make_test_case(
                1,
                5,
                7,
                rewards=[[-4.0, -1.0, 0.0, 2.0, 5.0]],
                clip_eps=0.1,
            )
        )

        # Zero KL difference isolates the PPO surrogate.
        tests.append(
            self._make_test_case(
                2,
                4,
                16,
                log_ref=[[[-0.2] * 16] * 4] * 2,
                log_pi=[[[-0.2] * 16] * 4] * 2,
            )
        )

        # Power-of-two group and sequence dimensions.
        tests.append(self._make_test_case(4, 8, 64, clip_eps=0.1, beta=0.05))

        # Non-power-of-two dimensions.
        tests.append(self._make_test_case(3, 5, 27))

        # Maximum group width exercises the group reduction boundary.
        tests.append(self._make_test_case(2, 32, 3))

        # Extreme but finite KL differences.
        tests.append(
            self._make_test_case(
                2,
                4,
                8,
                log_pi=[[[8.0] * 8] * 4] * 2,
                log_ref=[[[-8.0] * 8] * 4] * 2,
            )
        )

        # Large positive, finite KL log-ratio exercises the exponential branch.
        tests.append(
            self._make_test_case(
                2,
                4,
                8,
                log_pi=[[[-8.0] * 8] * 4] * 2,
                log_ref=[[[8.0] * 8] * 4] * 2,
            )
        )

        # Realistic rollout shape.
        tests.append(self._make_test_case(8, 8, 256, beta=0.02))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        B, G, S = 64, 16, 4096
        return {
            "rewards": RandnTensor((B, G)),
            "log_pi": RandnTensor((B, G, S)),
            "log_pi_old": RandnTensor((B, G, S)),
            "log_ref": RandnTensor((B, G, S)),
            "output": OutTensor((1,)),
            "clip_eps": 0.2,
            "beta": 0.01,
            "B": B,
            "G": G,
            "S": S,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_grpo_surrogate_loss(...):
#     ...


def solve(
    rewards: torch.Tensor,
    log_pi: torch.Tensor,
    log_pi_old: torch.Tensor,
    log_ref: torch.Tensor,
    output: torch.Tensor,
    clip_eps: float,
    beta: float,
    B: int,
    G: int,
    S: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
