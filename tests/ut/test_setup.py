# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SETUP_PATH = _REPOSITORY_ROOT / "setup.py"


class _Logger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def _load_compute_num_jobs():
    tree = ast.parse(_SETUP_PATH.read_text(encoding="utf-8"), filename=str(_SETUP_PATH))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "cmake_build_ext":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "compute_num_jobs":
                    module = ast.fix_missing_locations(ast.Module(body=[child], type_ignores=[]))
                    namespace: dict[str, Any] = {}
                    exec(compile(module, str(_SETUP_PATH), "exec"), namespace)
                    return namespace["compute_num_jobs"]
    raise AssertionError("cmake_build_ext.compute_num_jobs was not found")


def _compute_num_jobs(max_jobs: str | None, monkeypatch: pytest.MonkeyPatch, affinity: int = 8):
    fake_os = SimpleNamespace(
        sched_getaffinity=lambda _pid: set(range(affinity)),
        cpu_count=lambda: 16,
    )
    function = _load_compute_num_jobs()
    monkeypatch.setitem(function.__globals__, "envs", SimpleNamespace(MAX_JOBS=max_jobs))
    monkeypatch.setitem(function.__globals__, "logger", _Logger())
    monkeypatch.setitem(function.__globals__, "os", fake_os)
    return function(object())


@pytest.mark.parametrize("max_jobs", ["", "-2", " 4 ", "not-a-number", "²"])
def test_compute_num_jobs_invalid_value_falls_back(
    max_jobs: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _compute_num_jobs(max_jobs, monkeypatch) == 8


@pytest.mark.parametrize(("max_jobs", "expected"), [("4", 4), ("0", 1), ("１２", 12)])
def test_compute_num_jobs_valid_decimal_value(
    max_jobs: str, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _compute_num_jobs(max_jobs, monkeypatch) == expected


def test_compute_num_jobs_unset_value_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _compute_num_jobs(None, monkeypatch) == 8


def test_compute_num_jobs_falls_back_when_affinity_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function = _load_compute_num_jobs()
    fake_os = SimpleNamespace(cpu_count=lambda: 16)
    monkeypatch.setitem(function.__globals__, "envs", SimpleNamespace(MAX_JOBS=""))
    monkeypatch.setitem(function.__globals__, "logger", _Logger())
    monkeypatch.setitem(function.__globals__, "os", fake_os)

    assert function(object()) == 16
