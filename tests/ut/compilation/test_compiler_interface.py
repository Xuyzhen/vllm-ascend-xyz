# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import sys
from hashlib import sha256
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import Mock

import torch_npu
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig


def test_ascend_compiler_factors_and_static_kernel_cache(monkeypatch, tmp_path):
    if "npugraph_ex" not in sys.modules and find_spec("npugraph_ex") is None:
        monkeypatch.setitem(sys.modules, "npugraph_ex._acl_concrete_graph.static_kernel", Mock())
    monkeypatch.setattr(torch_npu, "__version__", "test-version", raising=False)
    from vllm_ascend.compilation import compiler_interface

    vllm_config = VllmConfig()
    expected = {
        "torch_npu_version": "test-version",
        "enable_npugraph_ex": True,
        "enable_static_kernel": True,
        "enable_super_kernel": False,
    }
    ascend_compilation_config = SimpleNamespace(
        enable_npugraph_ex=True, enable_static_kernel=True, enable_super_kernel=False
    )
    monkeypatch.setattr(
        compiler_interface,
        "get_ascend_config",
        lambda: SimpleNamespace(ascend_compilation_config=ascend_compilation_config),
    )
    initialize_static_kernel = Mock()
    monkeypatch.setattr(compiler_interface, "_initialize_static_kernel_runtime", initialize_static_kernel)

    compiler = compiler_interface.AscendCompiler()
    legacy_hash = getattr(compiler, "compute_hash", None)
    if hasattr(CompilerInterface, "compute_hash"):
        assert legacy_hash is not None
        assert legacy_hash(vllm_config) == sha256(str(expected).encode(), usedforsecurity=False).hexdigest()[:10]
    else:
        assert legacy_hash is None
        assert compiler.compile_factors(vllm_config) == expected

    assert compiler.vllm_config is vllm_config
    compiler.initialize_cache(str(tmp_path))
    initialize_static_kernel.assert_called_once_with(vllm_config)
