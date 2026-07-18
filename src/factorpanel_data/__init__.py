"""Causal, versioned data preparation for FactorPanel-FM."""

from .proxy_config import ProxyFactorConfig, load_proxy_config
from .proxy_store import ProxyFactorPanel, ProxyFactorStore
from .proxy_registry import (
    FACTOR_WINDOWS,
    PROXY_FACTOR_REGISTRY,
    PROXY_LABEL_REGISTRY,
    FactorDefinition,
    LabelDefinition,
    build_label_registry,
    build_proxy_factor_registry,
)

__all__ = [
    "FACTOR_WINDOWS",
    "PROXY_FACTOR_REGISTRY",
    "PROXY_LABEL_REGISTRY",
    "FactorDefinition",
    "LabelDefinition",
    "ProxyFactorConfig",
    "ProxyFactorPanel",
    "ProxyFactorStore",
    "build_label_registry",
    "build_proxy_factor_registry",
    "load_proxy_config",
]
