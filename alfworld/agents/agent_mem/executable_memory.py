"""Hot-reloadable executable memory (code patches)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

_BUILTIN_ROOT = Path(__file__).parent / "default_executable"
_MODULE_NAMES = ("beliefs", "heuristics", "validators", "recovery", "tests")


class ExecutableMemoryStore:
    """
    Loads patchable Python modules (beliefs, heuristics, validators, recovery, tests).
    Agent applies feedback by writing/patching source and calling reload().
    """

    def __init__(self, root: Optional[Path] = None, copy_defaults: bool = True):
        self.root = Path(root) if root else _BUILTIN_ROOT
        self.copy_defaults = copy_defaults
        self._modules: Dict[str, ModuleType] = {}
        self.patch_log: List[Dict[str, Any]] = []
        self._ensure_root()
        self.reload()

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.copy_defaults and self.root.resolve() != _BUILTIN_ROOT.resolve():
            for name in _MODULE_NAMES:
                dst = self.root / f"{name}.py"
                src = _BUILTIN_ROOT / f"{name}.py"
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)

    def module_path(self, name: str) -> Path:
        return self.root / f"{name}.py"

    def read_module_source(self, name: str) -> str:
        path = self.module_path(name)
        if not path.exists():
            raise FileNotFoundError(path)
        return path.read_text(encoding="utf-8")

    def write_module_source(self, name: str, source: str) -> None:
        path = self.module_path(name)
        path.write_text(source, encoding="utf-8")
        self.patch_log.append({"type": "write_module", "module": name, "chars": len(source)})
        self.reload_module(name)

    def patch_function(
        self,
        module_name: str,
        function_name: str,
        new_function_source: str,
    ) -> None:
        """
        Replace a top-level function in a module by splicing new source.
        new_function_source should be a full 'def name(...):' block.
        """
        import ast

        source = self.read_module_source(module_name)
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)

        target = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                target = node
                break
        if target is None:
            # append new function
            indented = textwrap.indent(new_function_source.strip() + "\n", "")
            new_source = source.rstrip() + "\n\n\n" + indented
        else:
            start, end = target.lineno - 1, target.end_lineno
            new_block = textwrap.dedent(new_function_source).strip() + "\n"
            if not new_block.endswith("\n"):
                new_block += "\n"
            lines[start:end] = [new_block]
            new_source = "".join(lines)

        self.write_module_source(module_name, new_source)

    def reload(self) -> None:
        for name in _MODULE_NAMES:
            path = self.module_path(name)
            if path.exists():
                self.reload_module(name)

    def reload_module(self, name: str) -> ModuleType:
        path = self.module_path(name)
        spec = importlib.util.spec_from_file_location(
            f"alfworld_agent_mem_{name}_{id(self)}", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        self._modules[name] = mod
        return mod

    def get(self, name: str) -> ModuleType:
        if name not in self._modules:
            self.reload_module(name)
        return self._modules[name]

    def call(self, module: str, func: str, *args, **kwargs) -> Any:
        mod = self.get(module)
        fn: Callable = getattr(mod, func)
        return fn(*args, **kwargs)

    def run_self_tests(self, context: Dict[str, Any]) -> List[str]:
        try:
            return self.call("tests", "run_self_tests", context)
        except Exception as exc:
            return [f"tests.run_self_tests crashed: {exc}"]

    def list_hooks(self) -> Dict[str, List[str]]:
        hooks = {
            "beliefs": ["update_beliefs"],
            "heuristics": ["rank_actions", "suggest_action"],
            "validators": ["validate_action"],
            "recovery": ["on_failure"],
            "tests": ["run_self_tests"],
        }
        return hooks
