"""Text observation parsers for agent working memory."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


def object_id_to_cls(object_hash: str) -> str:
    return object_hash.split()[0] if object_hash else ""


def parse_welcome_receptacles(obs: str) -> Dict[str, str]:
    """Parse room intro: 'you see a armchair 2, a diningtable 1, ...'."""
    if "you see" not in obs.lower():
        return {}
    after = re.split(r"you see", obs, flags=re.IGNORECASE)[-1]
    after = after.split("Your task is to", 1)[0]
    after = after.replace(" and a ", ", ").replace(" a ", ", ")
    chunk = after.strip(".,\n\r ")
    if not chunk:
        return {}
    parts = [p.strip() for p in chunk.split(",") if p.strip()]
    return {p: object_id_to_cls(p) for p in parts}


def parse_visible_objects(obs: str) -> Dict[str, str]:
    """Objects visible in a receptacle-centric observation."""
    if "you see nothing" in obs.lower():
        return {}
    if "you see" not in obs.lower():
        return {}
    obj_str = re.split(r"you see", obs, flags=re.IGNORECASE)[-1]
    obj_str = (
        obj_str.replace(" and a ", ", ")
        .replace(" a ", ", ")
        .split("Your task is to", 1)[0]
        .strip(".,\n\r ")
    )
    if not obj_str:
        return {}
    return {o.strip(): object_id_to_cls(o.strip()) for o in obj_str.split(",") if o.strip()}


def parse_current_receptacle(obs: str) -> Optional[str]:
    if "You open the" in obs:
        return " ".join(obs.split("You open the", 1)[-1].split()[:2]).strip(",.")
    if "is open." in obs:
        return " ".join(obs.split("is open.", 1)[0].split()[-2:]).strip(",.")
    if "On the" in obs:
        return " ".join(obs.split("On the", 1)[-1].split()[:2]).strip(",.")
    if "go to" in obs.lower():
        return None
    return None


def parse_task(obs: str) -> str:
    if "Your task is to:" in obs:
        return obs.partition("Your task is to: ")[-1].strip()
    return ""


def action_succeeded(feedback: str) -> bool:
    return "Nothing happens" not in feedback


def parse_inventory_from_action(action: str, feedback: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (item_id, source_recep) after a successful take."""
    if "take" not in action.lower() or not action_succeeded(feedback):
        return None, None
    parts = action.split()
    if len(parts) < 3:
        return None, None
    item = " ".join(parts[1:3])
    source = " ".join(parts[-2:]) if "from" in action else None
    return item, source


def parse_put_inventory(action: str, feedback: str) -> bool:
    if "put" in action.lower() or "move" in action.lower():
        return action_succeeded(feedback)
    return False


def classify_action(action: str) -> str:
    a = action.lower().strip()
    if a.startswith("go to"):
        return "goto"
    if a.startswith("open"):
        return "open"
    if a.startswith("close"):
        return "close"
    if a.startswith("take"):
        return "take"
    if a.startswith("put") or a.startswith("move"):
        return "put"
    if a.startswith("use") or a.startswith("toggle"):
        return "toggle"
    if a.startswith("heat") or a.startswith("cool") or a.startswith("clean"):
        return "transform"
    if a.startswith("look"):
        return "look"
    if a.startswith("inventory"):
        return "inventory"
    if a.startswith("examine"):
        return "examine"
    return "other"
