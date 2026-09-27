r"""
LeetGPU 118: Vision Transformer Patch Embedding
===============================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement the patch embedding stem of a Vision Transformer — the layer that turns a batch of
images into a sequence of tokens for ViT, CLIP, SigLIP and DiT. Given a batch of
`B` images of shape `(C, H, W)`, split every image into non-overlapping
`P × P` patches, project each flattened patch to a `D`-dimensional
embedding with `patch_weight` and `patch_bias`, prepend the learned
`cls_token`, and add the learned positional embedding `pos_embed`. All
tensors are `float32`, and `H` and `W` are always divisible by
`P`.

(diagram omitted, see challenge.html in leetgpu-challenges)

With \(g_h = H/P\), \(g_w = W/P\) and \(N = g_h \cdot g_w\) patches per image, patch index
\(n = p_y \cdot g_w + p_x\) covers image rows \(p_y P \ldots p_y P + P - 1\) and columns
\(p_x P \ldots p_x P + P - 1\). The embedding of that patch is

\[
\text{token}[b][n][d] = \text{patch_bias}[d] +
\sum_{c=0}^{C-1} \sum_{i=0}^{P-1} \sum_{j=0}^{P-1}
\text{images}[b][c][p_y P + i][p_x P + j] \cdot \text{patch_weight}[d][c][i][j]
\]

and the output sequence for image \(b\) is the CLS token followed by the \(N\) patch tokens,
with the positional embedding added to every row:

\[
\begin{aligned}
\text{output}[b][0][d] &= \text{cls_token}[d] + \text{pos_embed}[0][d] \\[4pt]
\text{output}[b][n+1][d] &= \text{token}[b][n][d] + \text{pos_embed}[n+1][d]
\end{aligned}
\]

Note that the CLS token is not projected — it is a learned vector copied into row 0 of
every sequence in the batch.

Implementation Requirements
---------------------------

- Implement the `solve` function with the signature unchanged.
- Do not use external libraries beyond the framework provided.
- Write the result into `output` in-place.

-
All tensors are contiguous and row-major: `images` is
`(B, C, H, W)`, `patch_weight` is `(D, C, P, P)`,
`patch_bias` and `cls_token` are `(D,)`,
`pos_embed` is `(N+1, D)`, and `output` is
`(B, N+1, D)`.

Example
-------

Input: `B` = 1, `C` = 1, `H` = 4, `W` = 4,
`P` = 2, `D` = 2 (so \(g_h = g_w = 2\) and \(N = 4\))

\(\text{images}[0][0]\) (\(4 \times 4\)):
\[
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16
\end{bmatrix}
\]
\(\text{patch_weight}\) (\(2\) filters of shape \(1 \times 2 \times 2\)) — filter 0 selects the
top-left pixel of a patch, filter 1 the bottom-right pixel:
\[
\text{patch_weight}[0][0] = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad
\text{patch_weight}[1][0] = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}
\]
\[
\text{patch_bias} = \begin{bmatrix} 0 & 0 \end{bmatrix}, \quad
\text{cls_token} = \begin{bmatrix} 0.5 & -0.5 \end{bmatrix}, \quad
\text{pos_embed} = \begin{bmatrix}
0 & 0 \\ 1 & -1 \\ 2 & -2 \\ 3 & -3 \\ 4 & -4
\end{bmatrix}
\]

The four patches are \(\begin{bmatrix} 1 & 2 \\ 5 & 6 \end{bmatrix}\),
\(\begin{bmatrix} 3 & 4 \\ 7 & 8 \end{bmatrix}\),
\(\begin{bmatrix} 9 & 10 \\ 13 & 14 \end{bmatrix}\) and
\(\begin{bmatrix} 11 & 12 \\ 15 & 16 \end{bmatrix}\), so the projected tokens are
\(\begin{bmatrix} 1 & 6 \end{bmatrix}\), \(\begin{bmatrix} 3 & 8 \end{bmatrix}\),
\(\begin{bmatrix} 9 & 14 \end{bmatrix}\) and \(\begin{bmatrix} 11 & 16 \end{bmatrix}\).

Output (\(5 \times 2\), after prepending the CLS token and adding `pos_embed`):
\[
\text{output}[0] = \begin{bmatrix}
0.5 & -0.5 \\
2 & 5 \\
5 & 6 \\
12 & 11 \\
15 & 12
\end{bmatrix}
\]

Constraints
-----------

- 1 ≤ `B` ≤ 128
- 1 ≤ `C` ≤ 4
- 1 ≤ `H`, `W` ≤ 1,024, both divisible by `P`
- 1 ≤ `P` ≤ 32
- 1 ≤ `D` ≤ 1,024
- All tensors are `float32` on the GPU
- Image values are in the range [-10, 10]

-
Performance is measured with `B` = 64, `C` = 3, `H` = 224,
`W` = 224, `P` = 16, `D` = 768

Run `python3 puzzles/leetgpu/118_vit_patch_embedding.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Vision Transformer Patch Embedding"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        images: torch.Tensor,
        patch_weight: torch.Tensor,
        patch_bias: torch.Tensor,
        cls_token: torch.Tensor,
        pos_embed: torch.Tensor,
        output: torch.Tensor,
        B: int,
        C: int,
        H: int,
        W: int,
        P: int,
        D: int,
    ):
        gh, gw = H // P, W // P
        N = gh * gw
        assert images.shape == (B, C, H, W)
        assert patch_weight.shape == (D, C, P, P)
        assert patch_bias.shape == (D,)
        assert cls_token.shape == (D,)
        assert pos_embed.shape == (N + 1, D)
        assert output.shape == (B, N + 1, D)
        assert (
            images.dtype
            == patch_weight.dtype
            == patch_bias.dtype
            == cls_token.dtype
            == pos_embed.dtype
            == output.dtype
            == torch.float32
        )

        # (B, C, gh, P, gw, P) -> (B, gh, gw, C, P, P) -> (B, N, C * P * P)
        patches = images.view(B, C, gh, P, gw, P).permute(0, 2, 4, 1, 3, 5).reshape(B, N, C * P * P)
        tokens = patches @ patch_weight.view(D, C * P * P).t() + patch_bias  # (B, N, D)
        cls = cls_token.view(1, 1, D).expand(B, 1, D)
        output.copy_(torch.cat([cls, tokens], dim=1) + pos_embed)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "images": (ctypes.POINTER(ctypes.c_float), "in"),
            "patch_weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "patch_bias": (ctypes.POINTER(ctypes.c_float), "in"),
            "cls_token": (ctypes.POINTER(ctypes.c_float), "in"),
            "pos_embed": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
            "H": (ctypes.c_int, "in"),
            "W": (ctypes.c_int, "in"),
            "P": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, B, C, H, W, P, D, fill=None):
        device = self.device
        dtype = torch.float32
        N = (H // P) * (W // P)
        if fill is None:
            images = torch.randn(B, C, H, W, device=device, dtype=dtype)
        else:
            images = torch.full((B, C, H, W), fill, device=device, dtype=dtype)
        return {
            "images": images,
            "patch_weight": torch.randn(D, C, P, P, device=device, dtype=dtype) * 0.02,
            "patch_bias": torch.randn(D, device=device, dtype=dtype) * 0.02,
            "cls_token": torch.randn(D, device=device, dtype=dtype) * 0.02,
            "pos_embed": torch.randn(N + 1, D, device=device, dtype=dtype) * 0.02,
            "output": torch.empty(B, N + 1, D, device=device, dtype=dtype),
            "B": B,
            "C": C,
            "H": H,
            "W": W,
            "P": P,
            "D": D,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        B, C, H, W, P, D = 1, 1, 4, 4, 2, 2
        images = torch.tensor(
            [
                [
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        # filter 0 selects the top-left pixel of a patch, filter 1 the bottom-right pixel
        patch_weight = torch.tensor(
            [[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 1.0]]]],
            device=device,
            dtype=dtype,
        )
        patch_bias = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        cls_token = torch.tensor([0.5, -0.5], device=device, dtype=dtype)
        pos_embed = torch.tensor(
            [[0.0, 0.0], [1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]],
            device=device,
            dtype=dtype,
        )
        return {
            "images": images,
            "patch_weight": patch_weight,
            "patch_bias": patch_bias,
            "cls_token": cls_token,
            "pos_embed": pos_embed,
            "output": torch.empty(B, 5, D, device=device, dtype=dtype),
            "B": B,
            "C": C,
            "H": H,
            "W": W,
            "P": P,
            "D": D,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: a single image that is exactly one patch
        tests.append(self._make_test_case(1, 1, 2, 2, 2, 4))

        # Edge case: tiny multi-channel image, 4 patches
        tests.append(self._make_test_case(2, 3, 4, 4, 2, 8))

        # Zero input: only bias, cls token and positional embedding survive
        tests.append(self._make_test_case(1, 3, 8, 8, 4, 16, fill=0.0))

        # Negative-only input
        tests.append(self._make_test_case(2, 3, 16, 16, 4, 32, fill=-1.5))

        # Power-of-2 sizes
        tests.append(self._make_test_case(4, 3, 32, 32, 16, 64))
        tests.append(self._make_test_case(2, 3, 64, 64, 8, 128))

        # Non-power-of-2 sizes
        tests.append(self._make_test_case(3, 3, 30, 30, 5, 100))

        # Non-square image with a non-power-of-2 patch size
        tests.append(self._make_test_case(5, 1, 18, 24, 6, 48))

        # Realistic: ViT-Tiny/16 at 224x224
        tests.append(self._make_test_case(8, 3, 224, 224, 16, 192))

        # Realistic: small patches, wide embedding
        tests.append(self._make_test_case(2, 3, 96, 96, 8, 256))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # ViT-Base/16 at 224x224 with a batch of 64 images
        torch.manual_seed(0)
        device = self.device
        dtype = torch.float32
        B, C, H, W, P, D = 64, 3, 224, 224, 16, 768
        N = (H // P) * (W // P)
        return {
            "images": torch.empty(B, C, H, W, device=device, dtype=dtype).uniform_(-1.0, 1.0),
            "patch_weight": torch.randn(D, C, P, P, device=device, dtype=dtype) * 0.02,
            "patch_bias": torch.randn(D, device=device, dtype=dtype) * 0.02,
            "cls_token": torch.randn(D, device=device, dtype=dtype) * 0.02,
            "pos_embed": torch.randn(N + 1, D, device=device, dtype=dtype) * 0.02,
            "output": torch.empty(B, N + 1, D, device=device, dtype=dtype),
            "B": B,
            "C": C,
            "H": H,
            "W": W,
            "P": P,
            "D": D,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_vit_patch_embedding(...):
#     ...


def solve(
    images: torch.Tensor,
    patch_weight: torch.Tensor,
    patch_bias: torch.Tensor,
    cls_token: torch.Tensor,
    pos_embed: torch.Tensor,
    output: torch.Tensor,
    B: int,
    C: int,
    H: int,
    W: int,
    P: int,
    D: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
