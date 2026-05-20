"""Belief updaters — patchable hooks run after each env step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from alfworld.agents.agent_mem.working_memory import WorkingMemory


def update_beliefs(
    wm: "WorkingMemory",
    feedback: str,
    last_action: str,
    admissible_commands: List[str],
    context: Dict[str, Any],
) -> None:
    """
    Refine working memory beliefs beyond the default parser.
    Agent can patch this function after failed searches / misleading feedback.
    """
    task = (wm.task or "").lower()

    # Infer subgoal from task type when not set
    if wm.current_subgoal is None and wm.step <= 2:
        if "heat" in task and "microwave" in task:
            wm.set_subgoal("find object → heat in microwave → place in target")
        elif "cool" in task and "fridge" in task:
            wm.set_subgoal("find object → cool in fridge → place in target")
        elif "clean" in task and "sink" in task:
            wm.set_subgoal("find object → clean at sink → place in target")
        elif "look at" in task and "lamp" in task:
            wm.set_subgoal("find object → toggle lamp → examine under light")
        elif "put" in task:
            wm.set_subgoal("find object → place in target receptacle")

    # Down-weight location belief for objects we tried to take but failed
    if "take" in last_action.lower() and not context.get("action_succeeded", True):
        parts = last_action.split()
        if len(parts) >= 3:
            obj = " ".join(parts[1:3])
            if obj in wm.object_beliefs:
                belief = wm.object_beliefs[obj]
                if wm.curr_recep in belief.hypotheses:
                    belief.hypotheses[wm.curr_recep] *= 0.3
                belief.hypotheses.setdefault("elsewhere", 0.5)
                wm.hidden_uncertainties.append(
                    f"Take failed for '{obj}' at {wm.curr_recep}; location belief weakened."
                )
