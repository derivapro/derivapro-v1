from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


class LazyImport:
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._load()(*args, **kwargs)


class LazyAttribute:
    def __init__(self, module_name: str, attribute_name: str) -> None:
        self.module_name = module_name
        self.attribute_name = attribute_name
        self._attribute: Any | None = None

    def _load(self) -> Any:
        if self._attribute is None:
            module = importlib.import_module(self.module_name)
            self._attribute = getattr(module, self.attribute_name)
        return self._attribute

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._load()(*args, **kwargs)
