"""Coordinates working memory and executable memory for one episode."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from alfworld.agents.agent_mem.executable_memory import ExecutableMemoryStore
from alfworld.agents.agent_mem.parsers import action_succeeded, parse_task
from alfworld.agents.agent_mem.working_memory import WorkingMemory


class MemoryController:
    """
    Per-episode memory facade used by LLM / scripted agents.

    Usage:
        mem = MemoryController(executable_root=Path("./my_agent_mem"))
        mem.reset(obs[0])
        ...
        mem.observe(feedback, last_action, admissible)
        action = mem.propose_action(admissible)  # heuristic + recovery
        mem.patch_function("heuristics", "rank_actions", new_src)  # after feedback
    """

    def __init__(
        self,
        executable_root: Optional[Path] = None,
        copy_defaults: bool = True,
        max_history: int = 50,
        max_failures: int = 10,
    ):
        self.working = WorkingMemory(max_history=max_history, max_failures=max_failures)
        self.executable = ExecutableMemoryStore(root=executable_root, copy_defaults=copy_defaults)
        self._last_action: str = ""
        self._last_feedback: str = ""
        self._test_errors: List[str] = []

    @property
    def test_errors(self) -> List[str]:
        return list(self._test_errors)

    def reset(self, initial_obs: str = "") -> None:
        task = parse_task(initial_obs) if initial_obs else ""
        self.working.reset(task=task)
        self._last_action = "restart"
        self._last_feedback = initial_obs
        if initial_obs:
            self.working.update(initial_obs, "restart", admissible_commands=[])

    def observe(
        self,
        feedback: str,
        last_action: str,
        admissible_commands: Optional[List[str]] = None,
        infos: Optional[Dict[str, Any]] = None,
    ) -> None:
        admissible = admissible_commands or []
        self._last_action = last_action
        self._last_feedback = feedback

        self.working.update(feedback, last_action, admissible_commands=admissible)

        ctx = self._build_context(infos)
        try:
            self.executable.call(
                "beliefs",
                "update_beliefs",
                self.working,
                feedback,
                last_action,
                admissible,
                ctx,
            )
        except Exception as exc:
            self.working.hidden_uncertainties.append(f"beliefs.update_beliefs error: {exc}")

        if not action_succeeded(feedback) and last_action not in ("", "restart"):
            try:
                recovery = self.executable.call(
                    "recovery",
                    "on_failure",
                    self.working,
                    last_action,
                    feedback,
                    admissible,
                    ctx,
                )
                if recovery:
                    self.working.set_subgoal(f"recovery: {recovery}")
            except Exception as exc:
                self.working.hidden_uncertainties.append(f"recovery.on_failure error: {exc}")

        self._test_errors = self.executable.run_self_tests(ctx)
        self.working._recompute_uncertainties()

    def _build_context(self, infos: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "working_memory": self.working,
            "action_succeeded": action_succeeded(self._last_feedback),
            "last_action": self._last_action,
            "infos": infos or {},
        }

    def validate_action(
        self, action: str, admissible_commands: List[str]
    ) -> Tuple[bool, Optional[str]]:
        try:
            return self.executable.call(
                "validators",
                "validate_action",
                self.working,
                action,
                admissible_commands,
                self._build_context(),
            )
        except Exception as exc:
            return True, None  # fail open

    def rank_actions(self, admissible_commands: List[str]) -> List[Tuple[str, float]]:
        try:
            return self.executable.call(
                "heuristics",
                "rank_actions",
                self.working,
                admissible_commands,
                self._build_context(),
            )
        except Exception:
            return [(c, 0.0) for c in admissible_commands]

    def suggest_action(self, admissible_commands: List[str]) -> Optional[str]:
        try:
            return self.executable.call(
                "heuristics",
                "suggest_action",
                self.working,
                admissible_commands,
                self._build_context(),
            )
        except Exception:
            return None

    def propose_action(
        self,
        admissible_commands: List[str],
        *,
        fallback_random: bool = False,
    ) -> str:
        """
        Pick an action using recovery hint → heuristic → first valid admissible.
        """
        import random

        if not admissible_commands:
            raise ValueError("empty admissible_commands")

        if not action_succeeded(self._last_feedback) and self._last_action not in ("", "restart"):
            try:
                rec = self.executable.call(
                    "recovery",
                    "on_failure",
                    self.working,
                    self._last_action,
                    self._last_feedback,
                    admissible_commands,
                    self._build_context(),
                )
                if rec and rec in admissible_commands:
                    ok, _ = self.validate_action(rec, admissible_commands)
                    if ok:
                        return rec
            except Exception:
                pass

        hint = self.suggest_action(admissible_commands)
        if hint:
            ok, _ = self.validate_action(hint, admissible_commands)
            if ok:
                return hint

        ranked = self.rank_actions(admissible_commands)
        for action, _score in ranked:
            ok, _ = self.validate_action(action, admissible_commands)
            if ok:
                return action

        if fallback_random:
            return random.choice(admissible_commands)
        return admissible_commands[0]

    def get_prompt_context(self) -> str:
        block = self.working.to_prompt_block()
        if self._test_errors:
            block += "\n\nExecutable memory self-test failures:\n"
            block += "\n".join(f"  - {e}" for e in self._test_errors)
        return block

    def snapshot(self) -> Dict[str, Any]:
        return {
            "working": self.working.snapshot(),
            "test_errors": self._test_errors,
            "executable_root": str(self.executable.root),
            "patch_log": list(self.executable.patch_log),
        }

    # --- patch API for agent ---

    def patch_module(self, module_name: str, source: str) -> None:
        self.executable.write_module_source(module_name, source)

    def patch_function(self, module_name: str, function_name: str, new_source: str) -> None:
        self.executable.patch_function(module_name, function_name, new_source)

    def read_executable(self, module_name: str) -> str:
        return self.executable.read_module_source(module_name)

    def reload_executable(self) -> None:
        self.executable.reload()
