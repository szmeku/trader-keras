Before you do anything say hello Simon.

Don't ignore any of the points in this whole document please.

## Workflow:
- **Always read `docs/` and `README.md` first** before exploring or modifying code — they are the primary source of truth for architecture, config, and conventions
- After every successful iteration, **update the relevant docs/** to keep them in sync with code changes
- Do **small, focused changes** — iterate repeatedly each time running tests so you get feedback from reality
- Before committing, **all tests must pass**
- **Commit at every iteration/integration step**
- Work through as many iterations until you have promising results as needed (don't bother user too often unless they are crucial doubts)
- After each iteration update STATUS.MD with progress of work
- Report: status, metrics
- When tasks involve bottleneck hardware (e.g., GPUs), ensure that no more than one agent per machine is assigned such a task at any given time. Running multiple agents that compete for the same hardware resource can lead to conflicts and unreliable results.
- remember to save your tokens if possible, you can run dumber agents to do simpler work, you have access to opencode cli where you can run "openrouter/moonshotai/kimi-k2.5" (i'm signed in)
- **Never read large log files whole** — always `tail`, `head`, or `wc -l` first. Logs (MT5 tester, training, etc.) can be 50k+ lines and will eat all context tokens. Use targeted searches (`grep`, `tail -c`) instead.
- After bigger changes update docs/ (use features like links between docs — user reads them in Obsidian)
- if on your way you notice/problem that is beyond your scope ands is not quick fix you should append it to docs/issues.md with all necessary details about the problem (and date_time of the report)
- **Remote training monitoring:** When running VastAI/remote training, check logs every ~1 minute (tail last few lines) to catch failures early. Don't sleep for 5-10 minutes between checks — errors like OOM, SSH failures, or preemptions should be detected quickly.
- **Use existing CLI tools first:** The CLI (`run.py`, `tools/`) is the **first place to go** for any operation. Only fall back to inline Python when the CLI genuinely cannot do what you need.
  - **Train**: `python run.py train config.yml`
  - **Re-evaluate**: `python run.py train eval_config.yml` (with `stage1.train: false` + `existing_wandb_run`)
  - **Anything the CLI can't do** → inline Python is acceptable (e.g., one-off debugging, ad-hoc queries)
- few agents can work on the same codebase, that's why for bigger tasks you have to work in separate worktree (subfolder) and do PR at the end

## Development Standards

**Size & Structure:**
- ~150-200 lines per file max
- High modularity — extract early, extract often
- Before creating any file, function, or module — **check what already exists**. Duplication is the cardinal sin.
- don't blow up the codebase

**Quality:**
- Clever simplicity — the smartest solution to a hard problem is a simple one
- Clean code principles; optimize for readability and maintainability
- Strict typing everywhere; run type checkers (`mypy` / `pyright`)
- Before adding a new dependency, check if an existing one already covers the need
- No ambiguity
- less ifs then better, make our exceptional cases as our standard cases (follow Linus Torvalds thinking in it)
- Never write by yourself something that is available as a library — good coders use others' solutions
- Default to finding generic patterns and shared abstractions; DRY aggressively
- use functional style programming when doesn't introduce computing/ram/resources overhead, simple straight forward flows, composability etc 
- when we already have some naming convention we should stick to it unless you figured out better one, ask user before the change 
- when possible single source of truth, repeating same stuff in many places makes it nigthmare to maintain

**Verification:**
- Write small, focused unit tests; run them every iteration
- Run sanity checks on small data samples before scaling up

**Architecture Decisions:**
- For non-trivial choices, append to `DECISIONS.md`:
  `## [Date] Title` → Context, Options, Decision, Reasoning
- Every decision is **provisional** — revisit and overturn when evidence
  or a better argument emerges. Nothing is sacred.

---

## Current Focus
- Current work is on Stage 1 (GRU predictor) and data pipeline

---

## Project Config — `trader`

| Item | Value |
|---|---|
| Environment | we have .venv and alias u that equals 'uv run' |
| Venv path | `~/projects/trader/.venv` |
| Data path | `~/projects/data` |

- torch is installed with good version, never change it
- All agents must activate this venv and use `uv` for dependency management.
