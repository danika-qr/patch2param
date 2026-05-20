"""LLM agent that patches executable memory (heuristics), not direct action selection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from alfworld.agents.agent_mem.llm_client import OpenAICompatibleClient
from alfworld.agents.agent_mem.memory_controller import MemoryController

PATCH_SYSTEM_PROMPT = """You maintain executable memory for an ALFWorld text agent.

You will receive:
- Working memory (beliefs, failures, subgoals, uncertainties)
- The current Python source of a hook function in heuristics/beliefs/recovery
- Why a patch is requested (e.g. action failed)

Your job: rewrite ONLY the target function so future steps behave better.
Do NOT choose actions yourself. Do NOT output shell commands.

Rules:
- Keep the same function name and compatible signature shown in the template.
- `wm` is WorkingMemory; `admissible_commands` is List[str]; `context` is Dict.
- `rank_actions` must return List[Tuple[str, float]] sorted high-to-low.
- `suggest_action` returns Optional[str].
- Use wm.recent_failures, wm.containers_searched, wm.inventory, wm.task, etc.
- Output JSON only: {"function_source": "def name(...):\\n    ..."}
"""


@dataclass
class PatchLog:
    episode: int
    step: int
    trigger: str
    module: str
    function: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    llm_response: str = ""
    applied: bool = False
    error: Optional[str] = None
    test_errors: List[str] = field(default_factory=list)


class LLMPatchAgent:
    """External LLM that edits executable memory; action selection stays in heuristics."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        *,
        patch_module: str = "heuristics",
        patch_function: str = "rank_actions",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        self.client = client
        self.patch_module = patch_module
        self.patch_function = patch_function
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logs: List[PatchLog] = []

    def build_patch_prompt(
        self,
        mem: MemoryController,
        *,
        trigger: str,
        last_action: str,
        last_feedback: str,
        admissible_commands: List[str],
    ) -> str:
        try:
            current_source = mem.read_executable(self.patch_module)
        except FileNotFoundError:
            current_source = "(module missing)"

        ranked_preview = ""
        try:
            ranked = mem.rank_actions(admissible_commands[:20])
            ranked_preview = "\n".join(f"  {a!r}: {s:.2f}" for a, s in ranked[:8])
        except Exception as exc:
            ranked_preview = f"(rank_actions error: {exc})"

        return "\n".join(
            [
                f"## Patch trigger\n{trigger}",
                "",
                "## Working memory\n",
                mem.get_prompt_context(),
                "",
                "## Last transition",
                f"Action: {last_action}",
                f"Feedback: {last_feedback[:600]}",
                "",
                "## Current heuristic ranking (top of admissible)",
                ranked_preview or "(none)",
                "",
                f"## Target: {self.patch_module}.{self.patch_function}",
                "Rewrite this function in the module below. Match signatures used by other hooks.",
                "",
                "```python",
                current_source,
                "```",
            ]
        )

    def request_patch(
        self,
        mem: MemoryController,
        *,
        episode: int,
        step: int,
        trigger: str,
        last_action: str,
        last_feedback: str,
        admissible_commands: List[str],
    ) -> PatchLog:
        log = PatchLog(
            episode=episode,
            step=step,
            trigger=trigger,
            module=self.patch_module,
            function=self.patch_function,
        )
        user_prompt = self.build_patch_prompt(
            mem,
            trigger=trigger,
            last_action=last_action,
            last_feedback=last_feedback,
            admissible_commands=admissible_commands,
        )
        messages = [
            {"role": "system", "content": PATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        log.messages = messages

        try:
            resp = self.client.chat(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            log.llm_response = resp.content
            if not (resp.content or "").strip():
                log.error = (
                    "empty LLM response (gpt-5 may need higher --max-tokens, e.g. 8192)"
                )
                self.logs.append(log)
                return log
            func_src = parse_function_source(resp.content)
            if not func_src:
                preview = resp.content[:300].replace("\n", "\\n")
                log.error = (
                    f"could not parse function_source from LLM response "
                    f"(preview: {preview}...)"
                )
                self.logs.append(log)
                return log

            if self.patch_function not in func_src:
                log.error = f"response missing def {self.patch_function}"
                self.logs.append(log)
                return log

            mem.patch_function(self.patch_module, self.patch_function, func_src)
            log.test_errors = mem.executable.run_self_tests(
                {"working_memory": mem.working}
            )
            log.applied = not log.test_errors
            if log.test_errors:
                log.error = "self_tests failed after patch"
        except Exception as exc:
            log.error = str(exc)

        self.logs.append(log)
        return log


def parse_function_source(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None

    # Whole response is JSON (common with gpt-5)
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "function_source" in data:
                return strip_code_fences(str(data["function_source"]))
        except json.JSONDecodeError:
            pass

    try:
        if "{" in text:
            blob = text[text.find("{") : text.rfind("}") + 1]
            data = json.loads(blob)
            if isinstance(data, dict) and "function_source" in data:
                return strip_code_fences(str(data["function_source"]))
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:python)?\s*(def\s+rank_actions\s*\(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"```(?:python)?\s*(def\s+\w+.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"(def\s+rank_actions\s*\(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    if text.startswith("def "):
        return strip_code_fences(text)
    return None


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:python)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()
