"""Dataset registry.

Each dataset module registers its PromptConfig via
``register()`` at import time.  The CLI resolves configs
by dataset name through ``get_prompt_config()``.
"""

from brv_bench.types import PromptConfig

_REGISTRY: dict[str, PromptConfig] = {}


def register(name: str, config: PromptConfig) -> None:
    """Register a dataset prompt config by name.

    Raises:
        ValueError: If *name* is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"Dataset '{name}' is already registered.")
    _REGISTRY[name] = config


def get_prompt_config(name: str) -> PromptConfig:
    """Look up the prompt config for a dataset by name.

    Raises:
        ValueError: If *name* is not registered.
    """
    config = _REGISTRY.get(name)
    if config is None:
        raise ValueError(
            f"No prompt config for dataset '{name}'. "
            f"Known datasets: {', '.join(sorted(_REGISTRY))}"
        )
    return config


def registered_datasets() -> list[str]:
    """Return sorted list of registered dataset names."""
    return sorted(_REGISTRY)
