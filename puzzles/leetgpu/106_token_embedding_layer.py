r"""
LeetGPU 106: Token Embedding Layer
==================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement the input embedding layer used at the start of transformer models such as BERT.
For each token in a batch, gather a row from the `token_embeddings` table using
`token_ids`, gather a row from the `position_embeddings` table using
`position_ids`, sum the two vectors, and apply Layer Normalization with learnable
scale and shift parameters along the embedding dimension.

Formally, for batch index \(b\) and time step \(t\), let
\[
s_{b,t} = E_T[\text{token\_ids}_{b,t}] + E_P[\text{position\_ids}_{t}] \in \mathbb{R}^{D}
\]
where \(E_T \in \mathbb{R}^{V \times D}\) and \(E_P \in \mathbb{R}^{P \times D}\) are the token
and positional embedding tables. The output is then
\[
\mu_{b,t} = \frac{1}{D} \sum_{d=1}^{D} s_{b,t,d}, \qquad
\sigma^2_{b,t} = \frac{1}{D} \sum_{d=1}^{D} (s_{b,t,d} - \mu_{b,t})^2,
\]
\[
y_{b,t,d} = \gamma_d \cdot \frac{s_{b,t,d} - \mu_{b,t}}{\sqrt{\sigma^2_{b,t} + \epsilon}} + \beta_d.
\]

Implementation Requirements
---------------------------

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- `token_ids` has shape `(B, T)` and `position_ids` has shape `(T)`, both with `int32` values
- `token_embeddings` has shape `(V, D)` and `position_embeddings` has shape `(P, D)`
- `gamma` and `beta` each have shape `(D)`
- The variance is computed without Bessel's correction (divide by `D`, not `D-1`)
- The final result must be stored in the `output` tensor with shape `(B, T, D)`

Example 1:
----------

Input:  B = 1, T = 2, V = 3, P = 2, D = 4, eps = 1e-5
token_ids        = [[2, 0]]
position_ids     = [0, 1]
token_embeddings = [[ 1.0,  2.0,  3.0,  4.0],
[ 0.0,  1.0,  0.0, -1.0],
[ 2.0,  0.0, -2.0,  0.0]]
position_embeddings = [[ 0.0,  0.0,  0.0,  0.0],
[ 1.0,  0.0, -1.0,  0.0]]
gamma = [1.0, 1.0, 1.0, 1.0]
beta  = [0.0, 0.0, 0.0, 0.0]
Output: output = [[[ 1.4142,  0.0000, -1.4142,  0.0000],
[-0.5773, -0.5773, -0.5773,  1.7320]]]

Constraints
-----------

- 1 ≤ `B` ≤ 64
- 1 ≤ `T` ≤ 1,024
- 1 ≤ `V` ≤ 50,000
- 1 ≤ `P` ≤ 4,096
- 1 ≤ `D` ≤ 1,024
- 0 ≤ `token_ids` values < `V`
- 0 ≤ `position_ids` values < `P`
- `eps` = 1e-5
- Performance is measured with `B` = 32, `T` = 512, `V` = 30,000, `P` = 2,048, `D` = 768

Run `python3 puzzles/leetgpu/106_token_embedding_layer.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Token Embedding Layer"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        position_embeddings: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        output: torch.Tensor,
        B: int,
        T: int,
        V: int,
        P: int,
        D: int,
        eps: float,
    ):
        assert token_ids.shape == (B, T)
        assert position_ids.shape == (T,)
        assert token_embeddings.shape == (V, D)
        assert position_embeddings.shape == (P, D)
        assert gamma.shape == (D,)
        assert beta.shape == (D,)
        assert output.shape == (B, T, D)
        assert token_ids.dtype == position_ids.dtype == torch.int32
        assert (
            token_embeddings.dtype
            == position_embeddings.dtype
            == gamma.dtype
            == beta.dtype
            == output.dtype
        )

        tok = token_embeddings[token_ids.long()]
        pos = position_embeddings[position_ids.long()]
        summed = tok + pos.unsqueeze(0)

        mean = summed.mean(dim=-1, keepdim=True)
        var = summed.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (summed - mean) * torch.rsqrt(var + eps)
        output.copy_(normalized * gamma + beta)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "token_ids": (ctypes.POINTER(ctypes.c_int), "in"),
            "position_ids": (ctypes.POINTER(ctypes.c_int), "in"),
            "token_embeddings": (ctypes.POINTER(ctypes.c_float), "in"),
            "position_embeddings": (ctypes.POINTER(ctypes.c_float), "in"),
            "gamma": (ctypes.POINTER(ctypes.c_float), "in"),
            "beta": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "T": (ctypes.c_int, "in"),
            "V": (ctypes.c_int, "in"),
            "P": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def _build_case(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        position_embeddings: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        B: int,
        T: int,
        V: int,
        P: int,
        D: int,
        eps: float,
    ) -> Dict[str, Any]:
        return {
            "token_ids": token_ids,
            "position_ids": position_ids,
            "token_embeddings": token_embeddings,
            "position_embeddings": position_embeddings,
            "gamma": gamma,
            "beta": beta,
            "output": torch.empty((B, T, D), device=self.device, dtype=torch.float32),
            "B": B,
            "T": T,
            "V": V,
            "P": P,
            "D": D,
            "eps": eps,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        idtype = torch.int32
        B, T, V, P, D = 1, 2, 3, 2, 4
        token_ids = torch.tensor([[2, 0]], device=self.device, dtype=idtype)
        position_ids = torch.tensor([0, 1], device=self.device, dtype=idtype)
        token_embeddings = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 0.0, -1.0],
                [2.0, 0.0, -2.0, 0.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        position_embeddings = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, -1.0, 0.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        gamma = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device, dtype=dtype)
        beta = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype)
        return self._build_case(
            token_ids,
            position_ids,
            token_embeddings,
            position_embeddings,
            gamma,
            beta,
            B,
            T,
            V,
            P,
            D,
            1e-5,
        )

    def _random_case(
        self,
        B: int,
        T: int,
        V: int,
        P: int,
        D: int,
        eps: float = 1e-5,
        emb_range: tuple = (-1.0, 1.0),
        gamma_range: tuple = (0.5, 1.5),
        beta_range: tuple = (-0.5, 0.5),
    ) -> Dict[str, Any]:
        dtype = torch.float32
        idtype = torch.int32
        token_ids = torch.randint(0, V, (B, T), device=self.device, dtype=idtype)
        position_ids = torch.randint(0, P, (T,), device=self.device, dtype=idtype)
        token_embeddings = torch.empty((V, D), device=self.device, dtype=dtype).uniform_(
            emb_range[0], emb_range[1]
        )
        position_embeddings = torch.empty((P, D), device=self.device, dtype=dtype).uniform_(
            emb_range[0], emb_range[1]
        )
        gamma = torch.empty((D,), device=self.device, dtype=dtype).uniform_(
            gamma_range[0], gamma_range[1]
        )
        beta = torch.empty((D,), device=self.device, dtype=dtype).uniform_(
            beta_range[0], beta_range[1]
        )
        return self._build_case(
            token_ids,
            position_ids,
            token_embeddings,
            position_embeddings,
            gamma,
            beta,
            B,
            T,
            V,
            P,
            D,
            eps,
        )

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        idtype = torch.int32
        tests: List[Dict[str, Any]] = []

        # tiny single-token case
        tests.append(
            self._build_case(
                torch.tensor([[0]], device=self.device, dtype=idtype),
                torch.tensor([0], device=self.device, dtype=idtype),
                torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=self.device, dtype=dtype),
                torch.tensor([[0.5, -0.5, 1.0, -1.0]], device=self.device, dtype=dtype),
                torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device, dtype=dtype),
                torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype),
                1,
                1,
                1,
                1,
                4,
                1e-5,
            )
        )

        # all-zero embeddings (degenerate variance handled by eps)
        B, T, V, P, D = 2, 3, 4, 3, 8
        tests.append(
            self._build_case(
                torch.randint(0, V, (B, T), device=self.device, dtype=idtype),
                torch.randint(0, P, (T,), device=self.device, dtype=idtype),
                torch.zeros((V, D), device=self.device, dtype=dtype),
                torch.zeros((P, D), device=self.device, dtype=dtype),
                torch.ones((D,), device=self.device, dtype=dtype),
                torch.zeros((D,), device=self.device, dtype=dtype),
                B,
                T,
                V,
                P,
                D,
                1e-5,
            )
        )

        # negative embeddings + non-trivial gamma/beta
        B, T, V, P, D = 2, 4, 5, 4, 8
        tests.append(
            self._build_case(
                torch.randint(0, V, (B, T), device=self.device, dtype=idtype),
                torch.tensor([0, 1, 2, 3], device=self.device, dtype=idtype),
                torch.empty((V, D), device=self.device, dtype=dtype).uniform_(-3.0, -0.5),
                torch.empty((P, D), device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                torch.empty((D,), device=self.device, dtype=dtype).uniform_(0.5, 2.0),
                torch.empty((D,), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                B,
                T,
                V,
                P,
                D,
                1e-5,
            )
        )

        # power-of-two dims, repeated token ids (heavy gather collisions)
        B, T, V, P, D = 4, 16, 8, 16, 32
        token_ids = torch.zeros((B, T), device=self.device, dtype=idtype)
        for b in range(B):
            token_ids[b] = torch.tensor([b % V] * T, device=self.device, dtype=idtype)
        tests.append(
            self._build_case(
                token_ids,
                torch.arange(T, device=self.device, dtype=idtype) % P,
                torch.empty((V, D), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                torch.empty((P, D), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                torch.empty((D,), device=self.device, dtype=dtype).uniform_(0.5, 1.5),
                torch.empty((D,), device=self.device, dtype=dtype).uniform_(-0.5, 0.5),
                B,
                T,
                V,
                P,
                D,
                1e-5,
            )
        )

        # power-of-two medium
        tests.append(self._random_case(B=8, T=64, V=256, P=128, D=64))

        # non-power-of-two
        tests.append(self._random_case(B=3, T=17, V=100, P=33, D=48))

        # larger non-power-of-two
        tests.append(self._random_case(B=5, T=100, V=1000, P=255, D=192))

        # realistic small transformer-like
        tests.append(self._random_case(B=4, T=128, V=2048, P=512, D=256))

        # realistic larger
        tests.append(self._random_case(B=8, T=256, V=5000, P=512, D=384, emb_range=(-0.3, 0.3)))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._random_case(
            B=32,
            T=512,
            V=30000,
            P=2048,
            D=768,
            eps=1e-5,
            emb_range=(-0.3, 0.3),
            gamma_range=(0.8, 1.2),
            beta_range=(-0.1, 0.1),
        )


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_token_embedding_layer(...):
#     ...


def solve(
    token_ids: torch.Tensor,
    position_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    position_embeddings: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    B: int,
    T: int,
    V: int,
    P: int,
    D: int,
    eps: float,
):
    # NOTE: the `T` argument shadows `tilelang.language as T` inside this function;
    # define your @tilelang.jit kernels at module level.
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
