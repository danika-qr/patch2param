"""Recovery rules after failures — patchable replanning hints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from alfworld.agents.agent_mem.working_memory import WorkingMemory


def on_failure(
    wm: "WorkingMemory",
    action: str,
    feedback: str,
    admissible_commands: List[str],
    context: Dict[str, Any],
) -> Optional[str]:
    """
    Suggest a recovery action after a failed step, or None.
    Patches here implement unstick / backoff / alternate receptacle search.
    """
    if not wm.recent_failures:
        return None

    last = wm.recent_failures[-1]
    low = action.lower()

    if low.startswith("open ") and "Nothing happens" in feedback:
        # Maybe already open — try look or take instead
        for cmd in admissible_commands:
            if cmd.lower().startswith("look"):
                return cmd

    if low.startswith("take "):
        # Search another receptacle of same class
        goto_cmds = [c for c in admissible_commands if c.lower().startswith("go to")]
        unsearched = [
            c
            for c in goto_cmds
            if c.replace("go to", "").strip() not in wm.containers_searched
        ]
        if unsearched:
            return unsearched[0]

    if low.startswith("go to"):
        wm.set_subgoal("recover: try different receptacle after blocked navigation")
        alts = [
            c
            for c in admissible_commands
            if c.lower().startswith("go to")
            and c != action
            and c.replace("go to", "").strip() not in wm.containers_searched
        ]
        if alts:
            return alts[0]

    return None
