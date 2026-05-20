#!/usr/bin/env python
"""
ALFWorld + agent memory + external LLM that PATCHES executable heuristics.

Architecture (intended):
  working memory.observe()  ← env feedback
  LLM reads mem             → patch heuristics.py (rank_actions / suggest_action)
  mem.propose_action()      → patched heuristics pick the action

NOT: LLM directly outputs admissible commands.

Examples:
  export OPENAI_API_KEY=sk-...
  python scripts/run_llm_agent_mem.py configs/eval_config.yaml \\
    --model gpt-4o-mini --mem-dir ./runs/exp1 --num-games 3

  python scripts/run_llm_agent_mem.py configs/eval_config.yaml \\
    --model Qwen2.5-7B-Instruct --base-url http://127.0.0.1:8000/v1 \\
    --mem-dir ./runs/exp1 --patch-on failure --patch-function rank_actions \\
    --dump ./runs/exp1/ep.jsonl --llm-log ./runs/exp1/patches.jsonl -v \\
    --log-file ./runs/exp1/terminal.log
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import IO, Optional, TextIO

from alfworld.agents.agent_mem import MemoryController
from alfworld.agents.agent_mem.config_loader import load_config_file, get_max_steps_per_episode
from alfworld.agents.agent_mem.llm_client import OpenAICompatibleClient
from alfworld.agents.agent_mem.llm_patch_agent import LLMPatchAgent
from alfworld.agents.environment import get_environment


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ALFWorld: mem.observe + LLM patch heuristics + heuristics act"
    )
    p.add_argument("config")
    p.add_argument("-p", "--params", nargs="+", default=[], metavar="key=value")

    p.add_argument("--num-games", type=int, default=1)
    p.add_argument(
        "--data-split",
        choices=("train", "valid_seen", "valid_unseen"),
        default="train",
        help="which game pool to sample: train (default) or valid_seen / valid_unseen (benchmark)",
    )
    # backward-compatible alias
    p.add_argument("--eval-split", choices=("valid_seen", "valid_unseen"), default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--mem-dir",
        type=str,
        default=None,
        help="REQUIRED for LLM runs: writable copy of executable memory (heuristics.py)",
    )
    p.add_argument("--dump", type=str, default=None)
    p.add_argument("--llm-log", type=str, default=None, help="JSONL of LLM patch requests/responses")

    p.add_argument("--model", required=True, help="LLM that patches executable memory")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="max_completion_tokens for patch LLM (gpt-5 needs headroom)",
    )

    p.add_argument(
        "--patch-on",
        choices=("failure", "episode_start", "step", "never"),
        default="failure",
        help="when to call LLM to rewrite heuristics: failure | episode_start | step (every step) | never",
    )
    p.add_argument("--patch-module", default="heuristics")
    p.add_argument("--patch-function", default="rank_actions")
    p.add_argument(
        "--patch-at-step0",
        action="store_true",
        help="also patch once at episode start (in addition to --patch-on)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="append all stdout prints to this file (terminal still shows output)",
    )
    return p


class _Tee(TextIO):
    """Write to terminal and a log file."""

    def __init__(self, *streams: IO[str]):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def _setup_log_file(path: str) -> IO[str]:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return open(log_path, "a", encoding="utf-8")


_DATA_SPLIT_MAP = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}


def _ensure_parent_dirs(*paths: Optional[str]) -> None:
    for p in paths:
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)


def resolve_mem_dir(mem_dir: Optional[str]) -> Path:
    if mem_dir:
        root = Path(mem_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root
    print(
        "Warning: --mem-dir not set; using a temp dir (patches discarded after exit). "
        "Set --mem-dir ./runs/xxx to keep heuristics across runs.",
        file=sys.stderr,
    )
    return Path(tempfile.mkdtemp(prefix="alfworld_agent_mem_"))


def should_patch(
    patch_on: str,
    *,
    at_episode_start: bool,
    step: int,
    last_feedback: str,
    patch_at_step0: bool,
) -> tuple[bool, str]:
    if patch_on == "never" and not (patch_at_step0 and step == 0):
        return False, ""
    if patch_at_step0 and step == 0:
        return True, "episode_start (patch-at-step0)"
    if patch_on == "episode_start" and step == 0:
        return True, "episode_start"
    if patch_on == "step":
        return True, "every_step"
    if patch_on == "failure" and "Nothing happens" in last_feedback:
        return True, "action_failed"
    return False, ""


def main() -> None:
    args = build_arg_parser().parse_args()

    log_f: Optional[IO[str]] = None
    orig_stdout = sys.stdout
    if args.log_file:
        log_f = _setup_log_file(args.log_file)
        sys.stdout = _Tee(orig_stdout, log_f)
        print(f"Logging stdout to {args.log_file}")

    try:
        _run(args)
    finally:
        sys.stdout = orig_stdout
        if log_f:
            log_f.close()


def _resolve_data_split(args: argparse.Namespace) -> str:
    if args.eval_split is not None:
        if args.data_split != "train":
            print("Warning: both --data-split and --eval-split set; using --eval-split", file=sys.stderr)
        return args.eval_split
    return args.data_split


def _run(args: argparse.Namespace) -> None:
    config = load_config_file(args.config, overrides=args.params)
    data_split = _resolve_data_split(args)
    train_eval = _DATA_SPLIT_MAP[data_split]
    _ensure_parent_dirs(args.dump, args.llm_log, args.log_file)

    env = get_environment(config["env"]["type"])(config, train_eval=train_eval)
    env = env.init_env(batch_size=1)

    mem_root = resolve_mem_dir(args.mem_dir)
    mem = MemoryController(executable_root=mem_root, copy_defaults=True)

    client = OpenAICompatibleClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    patcher = LLMPatchAgent(
        client,
        patch_module=args.patch_module,
        patch_function=args.patch_function,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    dump_f = open(args.dump, "a", encoding="utf-8") if args.dump else None
    llm_log_f = open(args.llm_log, "a", encoding="utf-8") if args.llm_log else None

    wins = 0
    max_steps = get_max_steps_per_episode(config)

    print(f"LLM patch model: {args.model}")
    print(f"Executable memory: {mem_root}")
    print(f"Action selection: mem.propose_action() → {args.patch_module}.{args.patch_function}")
    print(f"Patch when: {args.patch_on}" + (" + step0" if args.patch_at_step0 else ""))
    print(f"Data split: {data_split} (train_eval={train_eval})")

    for ep in range(args.num_games):
        obs, infos = env.reset()
        feedback = obs[0]
        mem.reset(feedback)
        admissible = list(infos["admissible_commands"][0])
        patcher.logs.clear()

        done = False
        step = 0
        last_action = "restart"

        if args.verbose:
            print(f"\n=== Episode {ep + 1} ===")
            print(mem.get_prompt_context())

        while not done and step < max_steps:
            do_patch, trigger = should_patch(
                args.patch_on,
                at_episode_start=True,
                step=step,
                last_feedback=feedback,
                patch_at_step0=args.patch_at_step0,
            )
            if do_patch:
                plog = patcher.request_patch(
                    mem,
                    episode=ep + 1,
                    step=step,
                    trigger=trigger,
                    last_action=last_action,
                    last_feedback=feedback,
                    admissible_commands=admissible,
                )
                if llm_log_f:
                    llm_log_f.write(json.dumps(asdict(plog), ensure_ascii=False) + "\n")
                if args.verbose:
                    status = "applied" if plog.applied else f"skipped ({plog.error})"
                    print(f"  [patch] {trigger} -> {status}")

            action = mem.propose_action(admissible, fallback_random=True)
            ok, reason = mem.validate_action(action, admissible)
            if not ok:
                action = mem.suggest_action(admissible) or admissible[0]

            obs, _, dones, infos = env.step([action])
            feedback = obs[0]
            done = bool(dones[0])
            admissible = list(infos["admissible_commands"][0])
            mem.observe(feedback, action, admissible, infos=infos)
            last_action = action
            step += 1

            if args.verbose or step % 10 == 0 or done:
                flag = "FAIL" if "Nothing happens" in feedback else "OK"
                print(f"  [{step}] {flag} | {action[:70]}")

        wins += int(infos["won"][0])
        snap = mem.snapshot()
        snap["won"] = bool(infos["won"][0])
        snap["steps"] = step
        snap["patch_model"] = args.model
        snap["patches_applied"] = sum(1 for lg in patcher.logs if lg.applied)
        if dump_f:
            dump_f.write(json.dumps(snap, ensure_ascii=False) + "\n")
        print(f"Episode {ep + 1}: won={snap['won']} steps={step} patches={snap['patches_applied']}")

    print(f"\nWin rate: {wins}/{args.num_games}")
    if dump_f:
        dump_f.close()
    if llm_log_f:
        llm_log_f.close()


if __name__ == "__main__":
    main()
