import torch
import torch_npu
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.rotary_embedding import rope_forward_oot
from vllm_ascend.ops.triton.muls_add import muls_add_triton
from vllm_ascend.utils import is_vl_model


def _get_ep_local_sizes(dp_metadata, ep_group) -> list[int] | None:
    """Return the SP token layout when the MoE runner installed it."""
    if dp_metadata is None:
        return None

    try:
        local_sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
    except (AssertionError, AttributeError):
        return None

    if local_sizes is None or len(local_sizes) != ep_group.world_size:
        return None
    return [int(size) for size in local_sizes]


def _pad_to_ep_local_size(x: torch.Tensor, max_local_size: int) -> torch.Tensor:
    """Make an EP all-gather input have the same first dimension on every rank."""
    if x.shape[0] == max_local_size:
        return x

    padded = x.new_zeros((max_local_size, *x.shape[1:]))
    copy_size = min(x.shape[0], max_local_size)
    padded[:copy_size].copy_(x[:copy_size])
    return padded


def _maybe_all_gather_and_maybe_unpad_impl(x: torch.Tensor) -> torch.Tensor:
    """仅用于 EP 通信场景：EP all_gather + 按 DP token 分布 unpad。

    v2 (rc1 sp_by_pass 还原): MoE 全程运行在 padded 布局上。输入 pad 到
    max_local_size 后 all_gather, 直接返回原始 gathered tensor (不做 unpad)。
    不平衡分片产生的多余 pad 行会被 MoE 计算后在 reduce 侧丢弃; 快 rank
    在 pad 行上浪费的 FLOPs 墙钟上免费 -- 它本来就要在集合通信处等慢 rank。
    换来的是完全消除 per-shard python 切片循环和 cat 的 host 开销
    (每 MoE 层 3 次 gather 调用, 61 层 prefill 热路径)。
    """
    forward_context = get_forward_context()
    dp_metadata = forward_context.dp_metadata
    ep_group = get_ep_group()
    local_sizes = _get_ep_local_sizes(dp_metadata, ep_group)
    if local_sizes is not None:
        max_local_size = max(local_sizes)
        # all_gather 要求各 rank 输入等长: pad 到 max_local_size。
        # 等分片时 _pad_to_ep_local_size 零开销原样返回。
        x = _pad_to_ep_local_size(x, max_local_size)
        # 直接返回 padded 布局 (rc1 sp_by_pass 语义), 不做 unpad。
        return ep_group.all_gather(x, 0).view(len(local_sizes) * max_local_size, *x.shape[1:])

    # need to unpad from ep size
    x = ep_group.all_gather(x, 0)
    if dp_metadata is not None:
        num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
        result = torch.empty((num_tokens_across_dp_cpu.sum(), *x.shape[1:]), device=x.device, dtype=x.dtype)
        dp_size = get_dp_group().world_size
        x = x.view(dp_size, _EXTRA_CTX.padded_length, *x.shape[1:])
        offset = 0
        for idx in range(dp_size):
            num_tokens_dp = int(num_tokens_across_dp_cpu[idx])
            result[offset : offset + num_tokens_dp] = x[idx, :num_tokens_dp]
            offset += num_tokens_dp
        x = result

    return x


def _maybe_pad_and_reduce_impl(x: torch.Tensor) -> torch.Tensor:
    """仅用于 EP 通信场景：按 DP token 分布 pad 后做 EP reduce_scatter。

    v2 (rc1 sp_by_pass 还原): gather 侧已保持 padded 布局 (每 rank 分片 =
    max_local_size, 真实 token 在前 pad 在后), reduce 输入天然就是
    (ep_size, max_local_size, ...) 的连续布局 -- 直接 view 后 reduce_scatter,
    零分配零拷贝, 输出 slice 回本 rank 真实 token 数即可。
    """
    forward_context = get_forward_context()

    if _EXTRA_CTX.is_draft_model and is_vl_model():
        return tensor_model_parallel_all_reduce(x)

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None:
        return get_ep_group().reduce_scatter(x, 0)

    ep_group = get_ep_group()
    local_sizes = _get_ep_local_sizes(dp_metadata, ep_group)
    if local_sizes is not None:
        # 与 v2 gather 配对: x 行数 = len(local_sizes) * max(local_sizes),
        # 每 rank 的真实 token 位于其分片头部, pad 尾部会被 slice 丢弃。
        reduced = ep_group.reduce_scatter(x.view(-1, *x.shape[1:]), 0)
        # The collective needs equal-sized chunks, while the next
        # sequence-parallel layer expects this rank's original token count.
        return reduced[: local_sizes[ep_group.rank_in_group]]

    # Pad each DP shard back to the common length before EP reduce-scatter.
    dp_size = get_dp_group().world_size
    num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
    padded_x = x.new_zeros((dp_size, _EXTRA_CTX.padded_length, *x.shape[1:]))
    offset = 0
    for idx in range(dp_size):
        num_tokens_dp = int(num_tokens_across_dp_cpu[idx])
        padded_x[idx, :num_tokens_dp] = x[offset : offset + num_tokens_dp]
        offset += num_tokens_dp

    return ep_group.reduce_scatter(padded_x.view(-1, *x.shape[1:]), 0)


def _maybe_all_gather_and_maybe_unpad_fake(x: torch.Tensor) -> torch.Tensor:
    forward_context = get_forward_context()
    ep_group = get_ep_group()
    local_sizes = _get_ep_local_sizes(forward_context.dp_metadata, ep_group)
    if local_sizes is not None:
        # 与 v2 gather impl 对齐: 输出为 padded 布局 (每分片 max_local_size)。
        return torch.empty((len(local_sizes) * max(local_sizes), *x.shape[1:]), device=x.device, dtype=x.dtype)

    return torch.empty((x.shape[0] * ep_group.world_size, *x.shape[1:]), device=x.device, dtype=x.dtype)


def _maybe_pad_and_reduce_fake(x: torch.Tensor) -> torch.Tensor:
    forward_context = get_forward_context()
    ep_group = get_ep_group()
    local_sizes = _get_ep_local_sizes(forward_context.dp_metadata, ep_group)
    if local_sizes is not None:
        return torch.empty(
            (local_sizes[ep_group.rank_in_group], *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )

    return torch.empty((x.shape[0] // ep_group.world_size, *x.shape[1:]), device=x.device, dtype=x.dtype)


# TODO(Angazenn): The reason why we use a custom op to encapsulate npu_quantize
# is that aclnnAscendQuantV3(npu_quantize) use div_mode=False, while
# aclnnAddRmsNormQuantV2(npu_add_rms_norm_quant) use div_moe=True. We have to
# pass input_scale and input_scale_reciprocal at the same time to avoid redundant
# reciprocal calculation in fussion pass. We shall remove this once
# aclnnAddRmsNormQuantV2 supports div_moe=False.
def _quantize_impl(
    in_tensor: torch.Tensor, input_scale: torch.Tensor, input_scale_reciprocal: torch.Tensor, input_offset: torch.Tensor
) -> torch.Tensor:
    return torch_npu.npu_quantize(in_tensor, input_scale_reciprocal, input_offset, torch.qint8, -1, False)


def _quantize_impl_fake(
    in_tensor: torch.Tensor, input_scale: torch.Tensor, input_scale_reciprocal: torch.Tensor, input_offset: torch.Tensor
) -> torch.Tensor:
    return torch_npu.npu_quantize(in_tensor, input_scale_reciprocal, input_offset, torch.qint8, -1, False)


def _rope_forward_oot_impl_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query, key


def _muls_add_impl_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="maybe_all_gather_and_maybe_unpad",
    op_func=_maybe_all_gather_and_maybe_unpad_impl,
    fake_impl=_maybe_all_gather_and_maybe_unpad_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_pad_and_reduce",
    op_func=_maybe_pad_and_reduce_impl,
    fake_impl=_maybe_pad_and_reduce_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="quantize",
    op_func=_quantize_impl,
    fake_impl=_quantize_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="npu_rotary_embedding",
    op_func=rope_forward_oot,
    fake_impl=_rope_forward_oot_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="muls_add",
    op_func=muls_add_triton,
    fake_impl=_muls_add_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
