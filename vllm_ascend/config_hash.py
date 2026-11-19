# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from typing import Any

from vllm.config import utils as config_utils


def hash_config(config: Any) -> str:
    """Compute a digest from the current configuration values."""
    upstream_hash = getattr(config_utils, "hash_config", None)
    if upstream_hash is None:
        return config.compute_hash()
    return upstream_hash(config)


def eplb_model_key(model_config: Any) -> str:
    """Look up a model with the key used by upstream EplbState.add_model."""
    upstream_hash_cached = getattr(config_utils, "hash_config_cached", None)
    if upstream_hash_cached is None:
        # Legacy EplbState.add_model registers with the uncached digest.
        return model_config.compute_hash()
    return upstream_hash_cached(model_config)
