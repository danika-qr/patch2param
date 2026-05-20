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
    Rank admissible actions with a strong bias toward finishing the search for a mug:
    - Prefer going to and opening unsearched cabinets over countertops and drawers.
    - Down-rank already searched containers and target receptacle (coffeemachine) before acquiring a mug.
    - Small penalties for redundant examine/close actions during the find phase.
    """
    scores: List[Tuple[str, float]] = []

    # Helpers
    def norm(text: str) -> str:
        return " ".join((text or "").lower().strip().split())

    def after(cmd: str, prefix: str) -> str:
        return norm(cmd[len(prefix):])

    def is_target_receptacle(text: str) -> bool:
        t = text.lower()
        return ("coffeemachine" in t) or ("coffee machine" in t) or ("coffee_maker" in t) or ("coffee maker" in t)

    def numeric_index_bias(name: str, kind_hint: str) -> float:
        # Prefer lower numeric indices slightly
        import re
        m = re.search(r"(\d+)", name)
        if not m:
            return 0.0
        try:
            idx = int(m.group(1))
        except Exception:
            return 0.0
        if "cabinet" in kind_hint:
            return max(0.0, 0.25 - 0.013 * idx)
        return max(0.0, 0.17 - 0.011 * idx)

    # Recent failures: avoid repeating exact failed actions
    failures = set()
    for f in getattr(wm, "recent_failures", []) or []:
        if isinstance(f, dict):
            a = f.get("action")
            if a:
                failures.add(a)
        elif isinstance(f, str):
            failures.add(f)

    # Containers searched (normalized)
    containers_searched_raw = getattr(wm, "containers_searched", set()) or set()
    if isinstance(containers_searched_raw, dict):
        containers_searched_raw = set(containers_searched_raw.keys())
    containers_searched = {norm(x) for x in containers_searched_raw}

    # Uncertainties -> set of container names (normalized)
    uncertainties_set: set = set()
    for u in getattr(wm, "uncertainties", []) or []:
        if isinstance(u, str):
            # Extract inside single quotes if present
            start = u.find("'")
            if start != -1:
                end = u.find("'", start + 1)
                if end != -1:
                    uncertainties_set.add(norm(u[start + 1 : end]))
                else:
                    uncertainties_set.add(norm(u))
            else:
                uncertainties_set.add(norm(u))
        elif isinstance(u, dict):
            name = u.get("name") or u.get("container") or u.get("object")
            if name:
                uncertainties_set.add(norm(name))

    # Task/subgoal context
    subgoal = (getattr(wm, "current_subgoal", None) or getattr(wm, "subgoal", "") or "")
    task = getattr(wm, "task", "") or ""
    inventory = getattr(wm, "inventory", []) or []

    need_mug = ("mug" in task.lower())
    have_mug = any("mug" in str(it).lower() for it in inventory)
    want_find = ("find" in subgoal.lower()) or (need_mug and not have_mug)

    # Detect if any open action here targets an unsearched container (discourage leaving)
    unsearched_openables_here = set()
    for c in admissible_commands:
        cl = norm(c)
        if cl.startswith("open "):
            tgt = after(c, "open ")
            if tgt and (tgt in uncertainties_set or tgt not in containers_searched):
                unsearched_openables_here.add(tgt)
    any_unsearched_openable_here = len(unsearched_openables_here) > 0

    # Track unsearched cabinets for strong boosting (esp. if only one remains)
    unsearched_cabinets = {u for u in uncertainties_set if "cabinet" in u}

    for cmd in admissible_commands:
        s = 0.0
        low = norm(cmd)

        if cmd in failures:
            s -= 2.0

        # Prefer to directly grab a visible mug
        if ("mug" in low) and (low.startswith("take ") or low.startswith("pick up") or "grab" in low):
            s += 3.0

        # OPEN actions (very valuable when searching)
        if low.startswith("open "):
            target = after(cmd, "open ")
            s += 1.2
            if want_find:
                s += 0.9
            if target in uncertainties_set:
                s += 1.0
            if target in containers_searched:
                s -= 0.6
            if need_mug and "cabinet" in target:
                s += 0.8
            if is_target_receptacle(target) and not have_mug:
                s -= 0.8

        # LOOK INSIDE actions (good right after opening)
        if low.startswith("look inside "):
            target = after(cmd, "look inside ")
            s += 0.6
            if want_find:
                s += 0.6
            if target in containers_searched:
                s -= 0.5

        # EXAMINE/LOOK AT: generally low value during search
        if low.startswith("examine ") or low.startswith("look at "):
            target = after(cmd, "examine ") if low.startswith("examine ") else after(cmd, "look at ")
            s -= 0.3
            if target in containers_searched:
                s -= 0.4

        # CLOSE: usually premature in find phase
        if want_find and low.startswith("close "):
            s -= 0.4

        # GO TO prioritization
        if low.startswith("go to "):
            target_raw = cmd[5:].strip()
            target = norm(target_raw)

            is_cab = "cabinet" in target
            is_draw = "drawer" in target
            is_counter = ("counter" in target or "countertop" in target)
            is_sink = ("sink" in target)

            # Base movement preference
            s += 0.1

            # Strong preference for unknown targets
            if target in uncertainties_set:
                s += 1.2
                if need_mug and is_cab:
                    s += 1.0
                elif need_mug and is_counter:
                    s += 0.6
                elif is_draw:
                    s += 0.15  # drawers are less promising
            else:
                # Penalize moving to already searched/known places
                if target in containers_searched:
                    s -= 1.0
                if need_mug and is_draw:
                    s -= 0.3

            # Extra boost if this is the last remaining unsearched cabinet
            if is_cab and target in unsearched_cabinets and len(unsearched_cabinets) == 1:
                s += 1.0

            # Avoid going to target receptacle before acquiring mug
            if not have_mug and is_target_receptacle(target):
                s -= 0.9

            # If there are unsearched openables here, discourage leaving
            if any_unsearched_openable_here:
                s -= 0.7

            # Visiting sink during find phase is premature
            if want_find and is_sink and not have_mug:
                s -= 0.35

            # Slight tie-break toward lower indices
            s += numeric_index_bias(target, target)

        # Tiny bias to check inventory occasionally
        if low.startswith("inventory"):
            s += 0.02

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
