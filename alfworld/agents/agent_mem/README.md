# Agent Memory (`alfworld.agents.agent_mem`)

Two-layer memory for agent-mem experiments on ALFWorld.

## Architecture (intended)

```
env.step → observe() → WorkingMemory (online state)
                ↓
         LLM reads mem + current heuristics.py
                ↓
         patch rank_actions / suggest_action  (executable memory)
                ↓
         mem.propose_action()  →  heuristics pick action
```

- **Working memory**: per-episode facts (inventory, locations, failures, uncertainty).
- **Executable memory**: patchable Python (`heuristics.py`, `beliefs.py`, …).
- **External LLM**: edits executable memory only; does **not** output `admissible_commands` directly.

## Scripts

| Script | LLM? | Who picks the action? |
|--------|------|---------------------|
| `scripts/run_agent_mem.py` | No | Random, or `--use-heuristic` (static default heuristics) |
| `scripts/run_llm_agent_mem.py` | Yes | **Always** `mem.propose_action()` after LLM may patch heuristics |

## CLI: `--mem-dir`, `--dump`, `--llm-log`

### `--mem-dir ./runs/exp1`

Persistent **workspace for executable memory** (Python files the LLM edits).

- First run: copies `default_executable/*.py` into this folder.
- LLM patches write here (`heuristics.py`, etc.); `reload` loads from here on the next step.
- **Working memory** stays in RAM only; it is **not** stored under `mem-dir`.
- Without `--mem-dir`, `run_llm_agent_mem.py` uses a **temp directory** (patches lost when the process exits).

Use one `mem-dir` per experiment so you can diff `heuristics.py` across runs or resume with the same policy code.

### `--dump ./runs/exp1/episodes.jsonl`

Append **one JSON line per episode** when the episode ends. Each line is `mem.snapshot()` plus `won`, `steps`, `patches_applied`, etc.

Contains structured **working memory** (inventory, beliefs, failures, …), not the full LLM transcript. Good for analysis / plotting success rate vs patch count.

### `--llm-log ./runs/exp1/patches.jsonl`

Append **one JSON line per LLM patch attempt** (not per env step).

Each line includes: trigger (`action_failed`, `episode_start`), prompt messages, raw LLM reply, whether the patch was applied, `self_tests` errors.

Does **not** log every step—only when the LLM is called to rewrite code. Use this to debug bad patches or replay prompts.

## LLM agent (`run_llm_agent_mem.py`)

```bash
export OPENAI_API_KEY=sk-...
python scripts/run_llm_agent_mem.py configs/eval_config.yaml \
  --model gpt-4o-mini \
  --mem-dir ./runs/exp1 \
  --patch-on failure \
  --dump ./runs/exp1/ep.jsonl \
  --llm-log ./runs/exp1/patches.jsonl \
  -v
```

| CLI | Meaning |
|-----|---------|
| `--model` | LLM that **patches** executable memory |
| `--mem-dir` | Where `heuristics.py` lives (strongly recommended) |
| `--patch-on failure` | Call LLM after `Nothing happens` (default) |
| `--patch-on episode_start` | Patch once at step 0 |
| `--patch-on step` | Call LLM **before every** `propose_action` |
| `--patch-on never` | No LLM patches (heuristics stay as on disk) |
| `--patch-at-step0` | Extra patch at episode start (in addition to `--patch-on`) |
| `--patch-module` / `--patch-function` | What to rewrite (default `heuristics.rank_actions`) |
| `--dump` | Episode-level memory snapshots |
| `--llm-log` | Per-patch LLM I/O |
| `--log-file` | Append all terminal stdout to a text file (see also `tee` below) |

ALFWorld env has no built-in actor/RM split. Optional RM for **patch quality** can be added later; action selection remains heuristics.

## Programmatic use

```python
from pathlib import Path
from alfworld.agents.agent_mem import MemoryController, LLMPatchAgent
from alfworld.agents.agent_mem.llm_client import OpenAICompatibleClient

mem = MemoryController(executable_root=Path("./runs/exp1"))
patcher = LLMPatchAgent(OpenAICompatibleClient(model="gpt-4o-mini"))

mem.reset(obs[0])
# after failure:
patcher.request_patch(mem, episode=1, step=t, trigger="action_failed", ...)
action = mem.propose_action(admissible)
env.step([action])
mem.observe(feedback, action, admissible)
```

Legacy `llm_agent.py` (`LLMActionAgent`) directly picks actions from the command list; kept for reference but **not** used by `run_llm_agent_mem.py`.
