# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm_ascend.ops import register_custom_ops as custom_ops


class _EpGroup:
    world_size = 4
    rank_in_group = 2
    max_local_size = 3

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 0
        assert x.shape == (3, 4)
        return torch.arange(48, dtype=x.dtype).view(12, 4)

    def reduce_scatter(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        # v2 (padded-layout) 语义: 输入已是 (ep_size, max_local_size) 布局,
        # 直接取本 rank 分片 (真实 token 在头部, pad 在尾部)。
        assert dim == 0
        assert x.shape == (12, 4)
        start = self.rank_in_group * self.max_local_size
        return x[start : start + self.max_local_size]


class _EpGroupRank0(_EpGroup):
    rank_in_group = 0


def _patch_sp_ep_context(monkeypatch):
    context = SimpleNamespace(
        dp_metadata=SimpleNamespace(
            get_chunk_sizes_across_dp_rank=lambda: [1, 1, 3, 3],
        ),
        is_draft_model=False,
    )
    monkeypatch.setattr(custom_ops, "_EXTRA_CTX", context)
    monkeypatch.setattr(custom_ops, "get_forward_context", lambda: context)
    monkeypatch.setattr(custom_ops, "get_ep_group", _EpGroup)


def test_sp_ep_all_gather_pads_and_unpads_local_chunks(monkeypatch):
    _patch_sp_ep_context(monkeypatch)

    result = custom_ops._maybe_all_gather_and_maybe_unpad_impl(torch.empty(1, 4))

    # v2 (padded-layout) 语义: 4 分片 x max(3) = 12 行, 不做 unpad。
    assert result.shape == (12, 4)
    assert torch.equal(result, torch.arange(48, dtype=result.dtype).view(12, 4))


def test_sp_ep_reduce_scatter_pads_local_chunks(monkeypatch):
    _patch_sp_ep_context(monkeypatch)

    result = custom_ops._maybe_pad_and_reduce_impl(torch.arange(48).view(12, 4))

    # rank2 分片 = 行 6..8, 全部为真实 token (local_size=3)。
    assert result.shape == (3, 4)
    assert torch.equal(result[:, 0], torch.tensor([24, 28, 32], dtype=result.dtype))


def test_sp_ep_reduce_scatter_unpads_local_chunk(monkeypatch):
    _patch_sp_ep_context(monkeypatch)
    monkeypatch.setattr(custom_ops, "get_ep_group", _EpGroupRank0)

    result = custom_ops._maybe_pad_and_reduce_impl(torch.arange(48).view(12, 4))

    # rank0 分片 = 行 0..2, 仅首行是真实 token (local_size=1), 尾部 pad 被 slice。
    assert result.shape == (1, 4)
    assert torch.equal(result[:, 0], torch.tensor([0], dtype=result.dtype))


def test_sp_ep_fake_shapes_follow_uneven_local_chunks(monkeypatch):
    _patch_sp_ep_context(monkeypatch)

    gathered = custom_ops._maybe_all_gather_and_maybe_unpad_fake(torch.empty(1, 4))
    reduced = custom_ops._maybe_pad_and_reduce_fake(torch.empty(12, 4))

    assert gathered.shape == (12, 4)
    assert reduced.shape == (3, 4)
