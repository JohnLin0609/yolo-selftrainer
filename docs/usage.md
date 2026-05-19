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
