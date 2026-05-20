"""Action validators — reject doomed commands before env.step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from alfworld.agents.agent_mem.working_memory import WorkingMemory


def validate_action(
    wm: "WorkingMemory",
    action: str,
    admissible_commands: List[str],
    context: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Return (ok, reason). If not ok, outer agent should pick another action.
    """
    if action not in admissible_commands:
        return False, "not in admissible_commands"

    for fail in wm.recent_failures:
        if fail["action"] == action and fail.get("location") == wm.curr_recep:
            return False, f"repeated failure at same location (step {fail['step']})"

    low = action.lower()
    if low.startswith("take ") and wm.inventory:
        return False, "inventory full"

    return True, None
