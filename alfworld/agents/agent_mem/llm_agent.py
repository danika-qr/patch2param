"""LLM policy: memory context -> admissible action string."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from alfworld.agents.agent_mem.llm_client import ActorRMClients, ChatResponse
from alfworld.agents.agent_mem.memory_controller import MemoryController


SYSTEM_PROMPT = """You are an embodied household agent in ALFWorld (text-based).
You receive structured working memory and a list of VALID commands.
You must output exactly ONE command copied verbatim from the valid list.
Reply with JSON only: {"action": "<command>"}"""


@dataclass
class LLMStepLog:
    step: int
    actor_messages: List[Dict[str, str]] = field(default_factory=list)
    actor_response: str = ""
    rm_messages: Optional[List[Dict[str, str]]] = None
    rm_response: Optional[str] = None
    rm_accepted: Optional[bool] = None
    parsed_action: str = ""
    fallback_used: bool = False


class LLMActionAgent:
    def __init__(
        self,
        clients: ActorRMClients,
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        use_rm: bool = False,
        rm_threshold: float = 6.0,
        max_rm_retries: int = 1,
    ):
        self.clients = clients
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_rm = use_rm and clients.reward_model is not None
        self.rm_threshold = rm_threshold
        self.max_rm_retries = max_rm_retries
        self.step_logs: List[LLMStepLog] = []

    def build_user_prompt(
        self,
        mem: MemoryController,
        feedback: str,
        admissible_commands: List[str],
    ) -> str:
        mem_block = mem.get_prompt_context()
        history = mem.working.observation_history[-3:]
        hist_lines = []
        for rec in history:
            status = "OK" if rec.succeeded else "FAIL"
            hist_lines.append(f"  [{status}] {rec.action} -> {rec.feedback[:120]}")

        numbered = "\n".join(f"  {i}. {cmd}" for i, cmd in enumerate(admissible_commands, 1))
        parts = [
            "## Working memory",
            mem_block,
            "",
            "## Latest environment feedback",
            feedback.strip() or "(none)",
        ]
        if hist_lines:
            parts += ["", "## Recent steps", *hist_lines]
        parts += [
            "",
            "## Valid commands (choose exactly one, copy verbatim)",
            numbered,
        ]
        return "\n".join(parts)

    def choose_action(
        self,
        mem: MemoryController,
        feedback: str,
        admissible_commands: List[str],
        *,
        step: int = 0,
    ) -> Tuple[str, LLMStepLog]:
        if not admissible_commands:
            raise ValueError("empty admissible_commands")

        log = LLMStepLog(step=step)
        user_prompt = self.build_user_prompt(mem, feedback, admissible_commands)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        log.actor_messages = messages

        action = None
        for attempt in range(1 + self.max_rm_retries):
            resp = self.clients.actor.chat(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            log.actor_response = resp.content
            action = parse_action_from_response(resp.content, admissible_commands)
            if action is None:
                action = fuzzy_match_action(resp.content, admissible_commands)
            if action is None:
                action = admissible_commands[0]
                log.fallback_used = True

            if not self.use_rm:
                break

            accepted, rm_log = self._rm_gate(
                mem, feedback, action, admissible_commands
            )
            log.rm_messages = rm_log.get("messages")
            log.rm_response = rm_log.get("response")
            log.rm_accepted = accepted
            if accepted:
                break
            messages.append({"role": "assistant", "content": resp.content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Reward model rejected action '{action}' "
                        f"(score below {self.rm_threshold}). "
                        "Pick a different valid command."
                    ),
                }
            )

        log.parsed_action = action
        self.step_logs.append(log)
        return action, log

    def _rm_gate(
        self,
        mem: MemoryController,
        feedback: str,
        action: str,
        admissible_commands: List[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        assert self.clients.reward_model is not None
        rm_prompt = "\n".join(
            [
                "Score whether this action is sensible given memory and feedback.",
                "Reply JSON only: {\"score\": <0-10 float>, \"accept\": <true|false>, \"reason\": \"...\"}",
                "",
                mem.get_prompt_context(),
                "",
                f"Latest feedback: {feedback[:500]}",
                f"Proposed action: {action}",
                "",
                "Valid commands:",
                *[f"  - {c}" for c in admissible_commands[:30]],
            ]
        )
        messages = [
            {
                "role": "system",
                "content": "You are a reward model for an ALFWorld agent. Be strict on invalid or repeated failures.",
            },
            {"role": "user", "content": rm_prompt},
        ]
        resp = self.clients.reward_model.chat(
            messages,
            temperature=0.0,
            max_tokens=256,
        )
        score, accept = parse_rm_response(resp.content)
        if accept is None:
            accept = score is not None and score >= self.rm_threshold
        return bool(accept), {"messages": messages, "response": resp.content}


def parse_action_from_response(text: str, admissible: List[str]) -> Optional[str]:
    text = text.strip()
    try:
        if "{" in text:
            blob = text[text.find("{") : text.rfind("}") + 1]
            data = json.loads(blob)
            if isinstance(data, dict) and "action" in data:
                return exact_or_none(str(data["action"]), admissible)
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        line = line.strip().strip("`\"'")
        if line.lower().startswith("action:"):
            return exact_or_none(line.split(":", 1)[1].strip(), admissible)
    return exact_or_none(text, admissible)


def exact_or_none(action: str, admissible: List[str]) -> Optional[str]:
    if action in admissible:
        return action
    norm = normalize_cmd(action)
    for cmd in admissible:
        if normalize_cmd(cmd) == norm:
            return cmd
    return None


def fuzzy_match_action(text: str, admissible: List[str]) -> Optional[str]:
    text_l = text.lower()
    best = None
    best_len = 0
    for cmd in admissible:
        if cmd.lower() in text_l and len(cmd) > best_len:
            best, best_len = cmd, len(cmd)
    return best


def normalize_cmd(cmd: str) -> str:
    return re.sub(r"\s+", " ", cmd.strip().lower())


def parse_rm_response(text: str) -> Tuple[Optional[float], Optional[bool]]:
    try:
        if "{" in text:
            blob = text[text.find("{") : text.rfind("}") + 1]
            data = json.loads(blob)
            score = float(data.get("score")) if "score" in data else None
            accept = data.get("accept")
            if isinstance(accept, bool):
                return score, accept
            if isinstance(accept, str):
                return score, accept.lower() in ("true", "yes", "1")
            return score, None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None, None
