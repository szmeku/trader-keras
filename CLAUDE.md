Before you do anything say hello Simon.

Don't ignore any of the points in this whole document please.

## Core Metric

**Maintainability is the primary quality metric.** Every decision — naming, structure, abstraction, dependency — should optimize for "how easy is this to understand and change 6 months from now?" Clever code that's hard to maintain is bad code.

## Workflow
- **Read `docs/` and `README.md` first** — they are the source of truth for architecture and conventions
- **Small, focused changes** — iterate, run tests each time, commit at every integration step
- All tests must pass before committing
- After each iteration update `STATUS.md`; after bigger changes update `docs/`
- Work through iterations autonomously — only ask user on crucial doubts
- If you notice a problem beyond scope, append to `docs/issues.md` with date and details
- For bigger tasks, work in a separate worktree and PR at the end
- Save tokens: use dumber agents (`opencode` with `openrouter/moonshotai/kimi-k2.5`) for simple work
- **Never read large log files whole** — use `tail`, `head`, `grep` first
- When monitoring remote training, check logs every ~1 minute
- One GPU-bound agent per machine at a time

## Development Standards

- **~150-200 lines per file max** — extract early, extract often
- **No duplication** — check what exists before creating anything
- **Minimal ifs** — make exceptional cases the standard case (Linus Torvalds style)
- **Single source of truth** — never repeat the same information in multiple places
- **Functional style** when it doesn't add overhead — composability, straight-forward flows
- **Strict typing everywhere** — run `mypy` / `pyright`
- Prefer libraries over hand-rolled solutions; check existing deps before adding new ones
- Stick to existing naming conventions unless you have a better one — ask before changing
- Write small, focused unit tests; sanity-check on small data before scaling
- Non-trivial architecture decisions go in `DECISIONS.md`: Context, Options, Decision, Reasoning

## CLI

- `python run.py` — train with default config (Hydra)
- `python run.py --config-name=bench` — use bench config
- `python run.py stage1.lr=0.001 data.load_limit=50000` — override params
- Disable W&B: `WANDB_MODE=disabled python run.py`

## Project Config

| Item | Value |
|---|---|
| Environment | `.venv` + alias `u` = `uv run` |
| Venv path | `~/projects/trader-keras/.venv` |
| Data path | `~/projects/data` |
| Backend | Keras 3 + JAX (no TensorFlow) |

- All agents must use `uv` for dependency management
- Current focus: Stage 1 (GRU predictor) and data pipeline
