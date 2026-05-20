"""Heuristic action ranking — patchable policies over admissible commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from alfworld.agents.agent_mem.working_memory import WorkingMemory


def rank_actions(
    wm: "WorkingMemory",
    admissible_commands: List[str],
    context: Dict[str, Any],
) -> List[Tuple[str, float]]:
    """
    Return (action, score) pairs; higher = preferred.
    Agent patches this to encode learned search order / recovery preferences.
    """
    scores: List[Tuple[str, float]] = []
    failures = {f["action"] for f in wm.recent_failures}

    for cmd in admissible_commands:
        s = 0.0
        low = cmd.lower()
        if cmd in failures:
            s -= 2.0
        if low.startswith("look"):
            s += 0.1
        if low.startswith("inventory"):
            s += 0.05
        if wm.current_subgoal and "find" in (wm.current_subgoal or "").lower():
            if low.startswith("go to"):
                recep = cmd.replace("go to", "").strip()
                if recep not in wm.containers_searched:
                    s += 0.5
        if wm.inventory and ("put" in (wm.current_subgoal or "").lower() or "place" in (wm.task or "").lower()):
            if low.startswith("move ") or low.startswith("put "):
                s += 1.0
        scores.append((cmd, s))

    scores.sort(key=lambda x: -x[1])
    return scores


def suggest_action(
    wm: "WorkingMemory",
    admissible_commands: List[str],
    context: Dict[str, Any],
) -> Optional[str]:
    """Return a single preferred action, or None to defer to the outer agent."""
    ranked = rank_actions(wm, admissible_commands, context)
    if not ranked:
        return None
    best_score = ranked[0][1]
    if best_score <= 0:
        return None
    return ranked[0][0]
