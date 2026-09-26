"""
Puzzle 07: Scalar FlashAttention
==============
From softmax to FlashAttention, we just need some computation.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
Now we have conquered softmax / online softmax, we can now implement one of the most important
operator in LLMs: FlashAttention.

To ensure a progressive learning experience, we will implement a scalar version of FlashAttention.
And we also remove the multi-head attention part. So in total we only have two dimensions: batch
size B and sequence length S, which are aligned with N, M in the previous puzzle. After such
simplification, you will find we are not so far from the FlashAttention algorithm. And with
TileLang, we can easily extend it to the full FlashAttention.

06-1: Simplified Scalar Flash Attention.

Inputs:
    Q: Tensor([B, S], float32)  # input tensor
    K: Tensor([B, S], float32)  # input tensor
    V: Tensor([B, S], float32)  # input tensor
    B: int   # batch size dimension. 1 <= B <= 256
    S: int   # sequence length dimension. 1 <= S <= 16384

Output:
    O: Tensor([B, S], float32)  # output tensor

Intermediates:
    MAX: float32  # max value of each row
    SUM: float32  # summation of each row
    QK: Tensor([B, S], float32)  # results of q*k
    P:  Tensor([B, S], float32)  # results of softmax(q*k) (not divided by summation).

Definition:
    for i in range(B):
        SUM = 0
        MAX = -inf
        for j in range(S):
            QK[i, j] = Q[i, j] * K[i, j]
            MAX = max(QK[i, j], MAX)
        for j in range(S):
            P[i, j] = exp(QK[i, j] - MAX)
            SUM += P[i, j]
        for j in range(M):
            O[i, j] = P[i, j] / SUM * V[i, j]
"""


def ref_scalar_flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    assert len(Q.shape) == 2
    assert len(K.shape) == 2
    assert len(V.shape) == 2
    assert Q.shape[0] == K.shape[0] == V.shape[0]  # B
    assert Q.shape[1] == K.shape[1] == V.shape[1]  # S
    assert Q.dtype == K.dtype == V.dtype == torch.float32
    return torch.softmax(Q * K, dim=1).mul_(V)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_scalar_flash_attn(Q, K, V, BLOCK_B: int, BLOCK_S: int):
    log2_e = 1.44269504
    B, S = T.const("B, S")
    dtype = T.float32
    Q: T.Tensor((B, S), dtype)
    K: T.Tensor((B, S), dtype)
    V: T.Tensor((B, S), dtype)
    O = T.empty((B, S), dtype)

    with T.Kernel(B, threads=256) as row:
        # 逻辑上，所有 threads 共享一个 BLOCK_S 的寄存器空间
        # 因为对 S 这个维度来说，我们都是先处理 (1, BLOCK_S) 个的局部元素，然后做全局的合并
        x = T.alloc_fragment((1, BLOCK_S), dtype)
        row_max = T.alloc_fragment((1,), dtype)
        row_sum = T.alloc_fragment((1,), dtype)

        T.fill(row_max, -T.infinity(dtype))
        T.clear(row_sum)

        for k in T.serial(T.ceildiv(S, BLOCK_S)):
            for j in T.Parallel(BLOCK_S):
                col = k * BLOCK_S + j
                x[0, j] = T.if_then_else(col < S, Q[row, col] * K[row, col], -T.infinity(dtype))

            # 保存上一个 [(k-1) * BLOCK_S, k * BLOCK_S) 的最大值
            row_max_old = row_max[0]
            # 计算新的最大值 [k * BLOCK_S, (k + 1) * BLOCK_S)
            T.reduce_max(x, row_max, dim=1, clear=False)

            # 修正并保存到 row_sum 中
            row_sum_old_fixed = row_sum[0] * T.exp2((row_max_old - row_max[0]) * log2_e)

            # 计算新的 BLOCK_S 的和
            for j in T.Parallel(BLOCK_S):
                x[0, j] = T.exp2((x[0, j] - row_max[0]) * log2_e)

            T.reduce_sum(
                x, row_sum, dim=1, clear=True
            )  # clear 必须要设置为 True，清除上一轮计算的结果

            # 两者加到一起
            row_sum[0] = row_sum_old_fixed + row_sum[0]

        # 我们已经得到了 MAX 和 SUM，重新计算 P[i, j]  = exp(Q[i, j] * K [i, j] - MAX)
        for k in T.serial(T.ceildiv(S, BLOCK_S)):
            for j in T.Parallel(BLOCK_S):
                col = k * BLOCK_S + j
                if col < S:
                    O[row, col] = (
                        V[row, col]
                        * T.exp2((Q[row, col] * K[row, col] - row_max[0]) * log2_e)
                        / row_sum[0]
                    )

    return O


def run_scalar_flash_attn():
    print("\n=== Scalar Flash Attention ===\n")
    B = 256
    S = 16384
    BLOCK_B = 16
    BLOCK_S = 128
    test_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
    )
    bench_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_scalar_flash_attn()
