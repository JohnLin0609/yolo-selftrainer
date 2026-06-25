# Usage guide

Day-to-day operations: dataset layout, API keys, monitoring, recovering from
HALTED, switching providers mid-run.

---

## Dataset layout

Two layouts work out of the box (the scaffold script auto-detects):

### A. Pre-split (Ultralytics standard)

```
my_dataset/
├── images/
│   ├── train/   *.jpg / *.png / *.bmp / *.tif
│   └── val/
└── labels/
    ├── train/   *.txt   (YOLO format: class cx cy w h)
    └── val/
```

### B. Flat with auto-split

```
my_dataset/
├── images/      # all images here (no train/val subdirs)
├── labels/      # all labels here
└── classes.txt  # one class name per line (optional, otherwise inferred)
```

`new_project.sh` will do an 80/20 split into `images/{train,val}` /
`labels/{train,val}` on the first run.

### Task auto-detection

The scaffold reads one label file and counts columns:
- **5 columns** → `detect` (class cx cy w h)
- **9 columns** → `obb` (class + 4 corner points)
- Polygon coordinates → `segment`
- Keypoint format → `pose`
- Directory-only (no labels) → `classify`

Override with `--task` if auto-detection guesses wrong.

---

## API keys

### Where to set them (pick one)

#### Method A — per-project `agent.env` (simplest for one-off projects)

After scaffolding in agent mode, edit `projects/<name>/agent.env`:

```bash
LLM_PROVIDER="openai"
LLM_MODEL="gpt-4o"

export OPENAI_API_KEY="sk-proj-..."
```

`start_agent.sh` sources this every round. **Already in `.gitignore`** — won't
be accidentally committed.

#### Method B — shell rc (one-time, works for all projects)

`~/.bashrc` or `~/.zshrc`:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
export GROQ_API_KEY="gsk_..."
```

#### Method C — central secrets file (recommended for long-term use)

```bash
mkdir -p ~/.config/yolo-selftrainer
cat > ~/.config/yolo-selftrainer/secrets.env <<'EOF'
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
EOF
chmod 600 ~/.config/yolo-selftrainer/secrets.env
```

Then add to the top of each project's `agent.env`:

```bash
[ -f ~/.config/yolo-selftrainer/secrets.env ] && source ~/.config/yolo-selftrainer/secrets.env
LLM_PROVIDER="..."
LLM_MODEL="..."
```

### Where to get the keys

| Provider | URL |
|---|---|
| Anthropic | https://console.anthropic.com/settings/keys |
| OpenAI | https://platform.openai.com/api-keys |
| Gemini | https://aistudio.google.com/apikey |
| Groq | https://console.groq.com/keys |
| Together | https://api.together.xyz/settings/api-keys |
| Ollama | none — runs locally |

### Verifying keys work

```bash
python3 scripts/test_agent_smoke.py --providers anthropic openai gemini
```

Each provider with a working key runs a 1-turn task to confirm tool calling
works end-to-end.

---

## Monitoring a running session

The training chain runs autonomously after launch. While it's running:

```bash
PROJECT=projects/my_project_name

# 1. Current state
echo "Round: $(cat $PROJECT/round_counter) / $(grep MAX_ROUNDS $PROJECT/start_claude.sh | head -1)"
echo "Failures: $(cat $PROJECT/consecutive_failures 2>/dev/null)"

# 2. Is training currently running?
PID=$(cat $PROJECT/train.pid 2>/dev/null)
kill -0 $PID 2>/dev/null && echo "running" || echo "between rounds"

# 3. Live training output (current epoch)
tail -f $PROJECT/current.log

# 4. Per-round metrics so far (P, R, mAP50, mAP50-95, losses, gap)
python3 scripts/event.py $PROJECT query metrics-table

# 5. Last events
tail -10 $PROJECT/events.jsonl | jq .

# 6. What the agent did most recently
tail -f $PROJECT/logs/*/round_*_pending/claude_session.log 2>/dev/null
```

---

## Recovering from HALTED

If you see a `HALTED` file in your project directory, the circuit breaker
tripped (3 consecutive failures). The chain is paused; no new training will
launch until you investigate.

```bash
PROJECT=projects/my_project

# 1. Read why
cat $PROJECT/HALTED

# 2. Inspect the failing run's logs
LAST_LOG=$(ls -td $PROJECT/logs/*/round_*_pending 2>/dev/null | head -1)
[ -z "$LAST_LOG" ] && LAST_LOG=$(ls -td $PROJECT/logs/*/2026* 2>/dev/null | head -1)
cat $LAST_LOG/claude_session.log
tail -50 $PROJECT/current.log

# 3. Fix the root cause (typically: bad params, OOM, dataset issue)

# 4. Resume
rm $PROJECT/HALTED $PROJECT/consecutive_failures
bash $PROJECT/start_claude.sh    # or start_agent.sh
```

### Plateau halts are different from crash halts

If `$PROJECT/HALTED` says `plateau circuit tripped`, this is **not a bug to
debug** — it means the framework noticed the primary metric stopped moving
for N+M rounds (defaults N=3, M=2) and stopped the loop cleanly so it
didn't burn the rest of your `--rounds` budget.

```bash
# Confirm it's a plateau halt
grep plateau $PROJECT/HALTED

# Inspect the trajectory the framework saw
python3 scripts/event.py $PROJECT query plateau-status
python3 scripts/event.py $PROJECT query metrics-table
```

You have three reasonable next steps:

1. **Collect more / cleaner data.** Plateau usually means you're at the
   dataset's information ceiling. More labelled examples or fixing noisy
   labels in low-performing classes is the right lever.
2. **Change architecture.** Swap model size, change task formulation, try
   a different pretrained backbone.
3. **Widen the plateau threshold.** If the val set is small and noisy and
   you're convinced the agent has more room:
   ```bash
   YOLO_TRAINER_PLATEAU_THRESHOLD=0.002 bash $PROJECT/start_claude.sh
   ```
   The query reads the env var, so the same threshold applies to both
   `train.sh`'s halt logic and `build_prompt.py`'s nudge. Other knobs:
   `YOLO_TRAINER_PLATEAU_N` (window size, default 3) and
   `YOLO_TRAINER_PLATEAU_M` (grace rounds after warning, default 2).

To resume: `rm $PROJECT/HALTED && bash $PROJECT/start_*.sh`.

---

## Switching providers mid-session

In agent mode, just edit `projects/<name>/agent.env` between rounds:

```bash
# Was using gpt-4o-mini for exploration; switch to claude for the home stretch
sed -i 's/^LLM_MODEL=.*/LLM_MODEL="claude-opus-4-7"/' projects/my_project/agent.env
sed -i 's/^LLM_PROVIDER=.*/LLM_PROVIDER="anthropic"/' projects/my_project/agent.env
```

The next round's wake-up will source the new config. No need to stop / restart
the chain — the change takes effect at the next `start_agent.sh` invocation
(triggered by `train.sh` finishing the current round).

---

## Stopping early

```bash
# Kill the running training (clean — won't wake next round)
PID=$(cat projects/my_project/train.pid)
kill $PID
rm projects/my_project/train.pid
```

Or just write a `HALTED` file manually:

```bash
echo "Operator stopped at $(date)" > projects/my_project/HALTED
# Then kill the current training process
kill $(cat projects/my_project/train.pid)
```

The chain is structured so killing the active `train.sh` is always safe —
worst case you lose the in-flight training run, but `events.jsonl` and all
completed rounds are intact.

---

## How the agent picks params

The agent does NOT edit `train.sh`. Each round it writes a flat JSON file:

```
projects/<name>/next_params.json
```

`scripts/apply_params.py` reads it, validates against `scripts/param_bounds.py`
(the single source of truth for bounds), and writes `effective_params.env`
that train.sh sources before launching `yolo train`. Out-of-range / unknown /
missing-required keys → `validation-failed` event emitted, `next_instruction.md`
rewritten with the violations, round NOT consumed.

To inspect the bounds that apply for a given task / fine-tune state:

```bash
python3 scripts/param_bounds.py show --task detect
python3 scripts/param_bounds.py show --task detect --fine-tune
python3 scripts/param_bounds.py show --task pose      # DEGREES capped at 15
```

Operator-side schema:

| Required key | Notes |
|---|---|
| `WEIGHTS` | Path to `.pt`. Ending in `/best.pt` flags it as a fine-tune. |
| `EPOCHS`, `PATIENCE`, `LR`, `LR_FINAL`, `IMGSZ`, `BATCH`, `OPTIMIZER` | See `BASE_BOUNDS` in `scripts/param_bounds.py`. |

Optional augmentation keys (`MOSAIC`, `MIXUP`, `COPY_PASTE`, `ERASING`,
`FLIPLR`, `FLIPUD`, `SCALE`, `DEGREES`, `WEIGHT_DECAY`, …) inherit their
scaffolded value when omitted from a round.

---

## Measuring agent uplift with `--mode baseline`

The agent (claude or litellm) decides hyperparameters each round. But is its
reasoning actually beating a dumb random sampler over the same parameter
space? Without a control, an mAP of 0.99 could be a real win, or just what
any policy inside the validator bounds would have produced.

`--mode baseline` runs the entire pipeline (preflight → train.sh → validator
→ events → circuit breaker → report) **identically**, except the per-round
hyperparameter choice comes from `scripts/baseline_policy.py` — a seeded
random search inside the validator bounds, with round 1 fixed to the
scaffolded defaults (the "no-tuning floor"). No LLM is called.

### How to run

```bash
# Default seed (42)
bash scripts/new_project.sh --dataset /path/to/dataset --max-rounds 10 \
    --mode baseline
cd projects/<name> && bash start_baseline.sh

# Pin the seed for explicit reproducibility (same seed → same trajectory)
bash scripts/new_project.sh --dataset /path/to/dataset --max-rounds 10 \
    --mode baseline --baseline-seed 7
cd projects/<name> && bash start_baseline.sh
```

`scripts/baseline_policy.py` prints KEY=VALUE pairs, which
`start_baseline.sh` sed-edits into `train.sh` and emits as a
`baseline-decision` event in `events.jsonl`. Unlike agent mode (which
goes through the `next_params.json` contract), baseline doesn't need the
validator — the policy samples directly from the validator bounds. From
train.sh's perspective the rest is identical: same circuit breaker, same
metric extraction, same held-out test eval, same plateau circuit.

### How to compare two reports

Run the agent and the baseline on the **same dataset, same rounds, same
`--test-seed`** so the held-out test split is identical. Both produce a
`projects/<name>/training_report.md`. Open them side-by-side and read three
fields:

| Field | Tells you |
|---|---|
| **Best Model** > primary metric | `uplift = mAP(agent) − mAP(baseline)` |
| **Held-out test evals** > best test value | uplift held on data the agent never saw |
| **Loop cost** > LLM cost / wall time | what the uplift cost in $$ + GPU time |

If `uplift < 0.02` (typical YOLO val noise floor), the agent isn't doing
anything random search wouldn't. If the test mAP doesn't move with val mAP,
the agent is fitting val noise — see also Boundary 4 in
[docs/architecture.md](architecture.md).

### Caveats

- **Variance on small val sets.** YOLO training is stochastic; identical
  runs can move mAP ±0.02 between training seeds. If observed uplift is in
  that noise floor, re-run baseline with 2–3 different `--baseline-seed`
  values and take the max.
- **Use the same `--test-split` and `--test-seed`.** Otherwise the test
  sets differ and the comparison is apples-to-oranges.
- **Round 1 = defaults.** Both modes start from the scaffolded params, so
  round 1 numbers should match exactly (modulo training stochasticity).
  Divergence appears from round 2 onward.
- **BATCH stays at `-1` in baseline.** Auto-batch avoids burning rounds on
  OOM exploration; if the agent is exploring batch sizes, that's an
  agent-side decision not modeled in the baseline.

---

## Where the best model lives

The trained weights are in `runs/<task>/<run_name>/weights/best.pt`.

The final round's `training_report.md` prints the full path of the best model
across all runs in the project (highest primary metric).

For deployment, copy that `best.pt` out:

```bash
BEST=$(grep -oE 'runs/[^"]+/best\.pt' projects/my_project/training_report.md | head -1)
cp "$BEST" /path/to/deployment/my_model.pt
```

---

## What NOT to do

- **Don't run two sessions on the same project concurrently.** State files
  (round_counter, events.jsonl) would race.
- **Don't manually edit `events.jsonl`.** It's the audit trail; corrupting
  it breaks `build_prompt.py` for subsequent rounds.
- **Don't commit `projects/`.** Already gitignored, but verify with
  `git status` before pushing.
- **Don't bypass the param validator.** If you find yourself wanting to
  set EPOCHS=2000, you probably don't — see the discussion in
  [docs/local-llms.md](local-llms.md) about why short rounds beat long ones
  on small datasets.
- **Don't `rm HALTED` and re-launch into a known plateau.** If the file
  says `plateau circuit tripped`, the framework already gave the agent
  N+M rounds to break through. Resuming without changing the dataset,
  model size, or `YOLO_TRAINER_PLATEAU_THRESHOLD` will just hit the same
  wall again. See "Recovering from HALTED → Plateau halts" above.
