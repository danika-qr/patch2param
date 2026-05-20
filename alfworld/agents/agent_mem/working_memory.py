"""Episodic working memory for a single ALFWorld task."""

from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, List, Optional, Set

from alfworld.agents.agent_mem import parsers


@dataclass
class ObservationRecord:
    step: int
    action: str
    feedback: str
    succeeded: bool
    location: Optional[str] = None


@dataclass
class ObjectBelief:
    """Belief over where an object instance may be (hidden-state uncertainty)."""

    object_id: str
    object_class: str
    hypotheses: Dict[str, float] = field(default_factory=dict)
    last_seen_at: Optional[str] = None
    last_seen_step: int = -1
    in_inventory: bool = False

    def entropy(self) -> float:
        import math

        probs = [p for p in self.hypotheses.values() if p > 0]
        if not probs:
            return 0.0
        return -sum(p * math.log(p + 1e-12) for p in probs)

    def most_likely_location(self) -> Optional[str]:
        if not self.hypotheses:
            return self.last_seen_at
        return max(self.hypotheses, key=self.hypotheses.get)


@dataclass
class ContainerBelief:
    receptacle_id: str
    receptacle_class: str
    opened: bool = False
    searched: bool = False
    may_contain: Set[str] = field(default_factory=set)
    uncertainty: float = 1.0  # 1 = unknown contents, 0 = fully observed empty/full


class WorkingMemory:
    """
    Online state for the current episode: history, inventory, spatial beliefs,
    container status, subgoal, failures, and hidden-state uncertainty.
    """

    def __init__(self, max_history: int = 50, max_failures: int = 10):
        self.max_history = max_history
        self.max_failures = max_failures
        self.reset()

    def reset(self, task: str = "") -> None:
        self.task: str = task
        self.step: int = 0
        self.observation_history: List[ObservationRecord] = []
        self.inventory: List[str] = []
        self.receptacles: Dict[str, str] = {}
        self.object_locations: Dict[str, str] = {}
        self.object_beliefs: Dict[str, ObjectBelief] = {}
        self.containers_opened: Set[str] = set()
        self.containers_searched: Set[str] = set()
        self.container_beliefs: Dict[str, ContainerBelief] = {}
        self.current_subgoal: Optional[str] = None
        self.subgoal_stack: List[str] = []
        self.recent_failures: Deque[Dict[str, Any]] = deque(maxlen=self.max_failures)
        self.curr_recep: str = ""
        self.obs_at_recep: Dict[str, str] = {}
        self.hidden_uncertainties: List[str] = []

    def snapshot(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "step": self.step,
            "inventory": list(self.inventory),
            "object_locations": dict(self.object_locations),
            "receptacles": dict(self.receptacles),
            "containers_opened": sorted(self.containers_opened),
            "containers_searched": sorted(self.containers_searched),
            "current_subgoal": self.current_subgoal,
            "subgoal_stack": list(self.subgoal_stack),
            "recent_failures": list(self.recent_failures),
            "curr_recep": self.curr_recep,
            "hidden_uncertainties": list(self.hidden_uncertainties),
            "object_beliefs": {
                k: {
                    "object_class": v.object_class,
                    "hypotheses": dict(v.hypotheses),
                    "last_seen_at": v.last_seen_at,
                    "entropy": v.entropy(),
                    "in_inventory": v.in_inventory,
                }
                for k, v in self.object_beliefs.items()
            },
            "observation_history": [asdict(r) for r in self.observation_history[-10:]],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.snapshot(), indent=indent)

    def to_prompt_block(self) -> str:
        """Compact natural-language summary for LLM / logging."""
        lines = [
            f"Task: {self.task}",
            f"Step: {self.step}",
            f"Subgoal: {self.current_subgoal or '(none)'}",
            f"Location: {self.curr_recep or '(unknown)'}",
            f"Inventory: {', '.join(self.inventory) if self.inventory else '(empty)'}",
        ]
        if self.object_locations:
            locs = [f"{o} @ {loc}" for o, loc in sorted(self.object_locations.items())]
            lines.append("Known objects: " + "; ".join(locs[:12]))
            if len(locs) > 12:
                lines.append(f"  ... +{len(locs) - 12} more")
        opened = sorted(self.containers_opened)
        searched = sorted(self.containers_searched)
        if opened:
            lines.append("Opened containers: " + ", ".join(opened))
        if searched:
            lines.append("Searched containers: " + ", ".join(searched))
        if self.recent_failures:
            fails = [f"{f['action']} (step {f['step']})" for f in list(self.recent_failures)[-3:]]
            lines.append("Recent failures: " + "; ".join(fails))
        if self.hidden_uncertainties:
            lines.append("Uncertainties:")
            lines.extend(f"  - {u}" for u in self.hidden_uncertainties[:8])
        return "\n".join(lines)

    def _ensure_container_belief(self, recep_id: str) -> ContainerBelief:
        if recep_id not in self.container_beliefs:
            cls = self.receptacles.get(recep_id, parsers.object_id_to_cls(recep_id))
            self.container_beliefs[recep_id] = ContainerBelief(
                receptacle_id=recep_id, receptacle_class=cls
            )
        return self.container_beliefs[recep_id]

    def _set_object_location(self, obj_id: str, location: str, step: int, confidence: float = 0.95) -> None:
        self.object_locations[obj_id] = location
        cls = parsers.object_id_to_cls(obj_id)
        if obj_id not in self.object_beliefs:
            self.object_beliefs[obj_id] = ObjectBelief(object_id=obj_id, object_class=cls)
        belief = self.object_beliefs[obj_id]
        belief.last_seen_at = location
        belief.last_seen_step = step
        belief.in_inventory = location == "agent"
        belief.hypotheses = {location: confidence}

    def _mark_container_opened(self, recep_id: str) -> None:
        self.containers_opened.add(recep_id)
        cb = self._ensure_container_belief(recep_id)
        cb.opened = True

    def _mark_container_searched(self, recep_id: str, visible_classes: Optional[Set[str]] = None) -> None:
        self.containers_searched.add(recep_id)
        cb = self._ensure_container_belief(recep_id)
        cb.searched = True
        cb.uncertainty = 0.2 if visible_classes else 0.0
        if visible_classes is not None:
            cb.may_contain = set(visible_classes)

    def _record_failure(self, action: str, feedback: str, reason: str = "") -> None:
        self.recent_failures.append(
            {
                "step": self.step,
                "action": action,
                "feedback": feedback[:200],
                "reason": reason or "Nothing happens",
                "location": self.curr_recep,
            }
        )

    def _recompute_uncertainties(self) -> None:
        notes: List[str] = []
        for recep_id, cb in self.container_beliefs.items():
            if recep_id in self.receptacles and recep_id not in self.containers_searched:
                notes.append(
                    f"Container '{recep_id}' ({cb.receptacle_class}) not searched; contents unknown."
                )
        for obj_id, belief in self.object_beliefs.items():
            if belief.in_inventory:
                continue
            if belief.entropy() > 0.5 and obj_id not in self.object_locations:
                loc = belief.most_likely_location()
                notes.append(
                    f"Object '{obj_id}' location uncertain (H={belief.entropy():.2f}"
                    + (f", best guess: {loc})" if loc else ")")
                )
        unseen_receps = set(self.receptacles) - self.containers_searched
        if unseen_receps and len(unseen_receps) <= 6:
            notes.append(f"Unsearched receptacles in scene: {', '.join(sorted(unseen_receps)[:6])}")
        elif unseen_receps:
            notes.append(f"{len(unseen_receps)} receptacles never visited/search incomplete.")
        self.hidden_uncertainties = notes

    def update(
        self,
        feedback: str,
        last_action: str,
        admissible_commands: Optional[List[str]] = None,
    ) -> None:
        """Apply one environment transition to working memory."""
        self.step += 1
        succeeded = parsers.action_succeeded(feedback)
        recep = parsers.parse_current_receptacle(feedback)

        record = ObservationRecord(
            step=self.step,
            action=last_action,
            feedback=feedback,
            succeeded=succeeded,
            location=recep or self.curr_recep,
        )
        self.observation_history.append(record)
        if len(self.observation_history) > self.max_history:
            self.observation_history = self.observation_history[-self.max_history :]

        if not succeeded and last_action not in ("restart", ""):
            self._record_failure(last_action, feedback)

        # Welcome / room scan
        if "Welcome" in feedback or (
            not self.receptacles and "you see" in feedback and "middle of a room" in feedback
        ):
            self.receptacles.update(parsers.parse_welcome_receptacles(feedback))
            for rid, rcls in self.receptacles.items():
                self._ensure_container_belief(rid)

        action_kind = parsers.classify_action(last_action)

        if action_kind == "goto" and succeeded:
            self.curr_recep = last_action.replace("go to", "").strip()
            recep = self.curr_recep
            if "open" in feedback.lower() or self.curr_recep not in self.containers_opened:
                self._mark_container_searched(self.curr_recep)

        if recep:
            self.curr_recep = recep

        visible = parsers.parse_visible_objects(feedback)
        if visible and self.curr_recep:
            self.obs_at_recep[self.curr_recep] = feedback
            classes = {cls for cls in visible.values()}
            self._mark_container_searched(self.curr_recep, classes)
            for obj_id, obj_cls in visible.items():
                self._set_object_location(obj_id, self.curr_recep, self.step)
                if obj_id not in self.object_beliefs:
                    self.object_beliefs[obj_id] = ObjectBelief(
                        object_id=obj_id, object_class=obj_cls
                    )

        if action_kind == "open" and succeeded:
            target = last_action.replace("open", "").strip()
            self._mark_container_opened(target)
            self.curr_recep = target

        if action_kind == "take" and succeeded:
            item, source = parsers.parse_inventory_from_action(last_action, feedback)
            if item:
                if item not in self.inventory:
                    self.inventory.append(item)
                src = source or self.curr_recep
                self._set_object_location(item, "agent", self.step)
                if src and item in self.object_locations:
                    pass
                elif src:
                    self._set_object_location(item, "agent", self.step)

        if action_kind == "put" and succeeded and self.inventory:
            item = self.inventory.pop()
            dest = self.curr_recep or "unknown"
            self._set_object_location(item, dest, self.step)

        if action_kind == "inventory" or "You are carrying:" in feedback:
            if "nothing" in feedback.lower():
                self.inventory = []
            elif "carrying:" in feedback.lower():
                part = feedback.split("carrying:", 1)[-1].strip().rstrip(".")
                if part:
                    self.inventory = [part]

        self._recompute_uncertainties()

    def set_subgoal(self, subgoal: str, push: bool = False) -> None:
        if push and self.current_subgoal:
            self.subgoal_stack.append(self.current_subgoal)
        self.current_subgoal = subgoal

    def pop_subgoal(self) -> Optional[str]:
        self.current_subgoal = self.subgoal_stack.pop() if self.subgoal_stack else None
        return self.current_subgoal

    def deep_copy(self) -> "WorkingMemory":
        wm = WorkingMemory(max_history=self.max_history, max_failures=self.max_failures)
        wm.task = self.task
        wm.step = self.step
        wm.observation_history = copy.deepcopy(self.observation_history)
        wm.inventory = list(self.inventory)
        wm.receptacles = dict(self.receptacles)
        wm.object_locations = dict(self.object_locations)
        wm.object_beliefs = copy.deepcopy(self.object_beliefs)
        wm.containers_opened = set(self.containers_opened)
        wm.containers_searched = set(self.containers_searched)
        wm.container_beliefs = copy.deepcopy(self.container_beliefs)
        wm.current_subgoal = self.current_subgoal
        wm.subgoal_stack = list(self.subgoal_stack)
        wm.recent_failures = deque(self.recent_failures, maxlen=self.max_failures)
        wm.curr_recep = self.curr_recep
        wm.obs_at_recep = dict(self.obs_at_recep)
        wm.hidden_uncertainties = list(self.hidden_uncertainties)
        return wm
