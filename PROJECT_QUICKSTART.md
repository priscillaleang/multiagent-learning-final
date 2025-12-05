# NFSP vs PSRO: Complete Project Quickstart

## Overview
This is a **complete step-by-step guide** to implementing and running Neural Fictitious Self-Play (NFSP) and Policy Space Response Oracles (PSRO) on 4 OpenSpiel games using TensorFlow and Google Cloud Platform.

**Estimated Timeline:** 4-5 weeks  
**Compute Required:** ~280 GPU-hours (~$50 on GCP with preemptible instances)  
**Skills Required:** Python, Deep RL, Game Theory basics

---

## Part 1: Local Development Setup (Days 1-3)

### 1.1 Repository Structure
```bash
mkdir -p multiagent-learning-project
cd multiagent-learning-project

# Create directory structure
mkdir -p src/{algorithms,networks,games,utils}
mkdir -p experiments/{configs,runners,gcp}
mkdir -p analysis/{plots,stats}
mkdir -p data/{raw,processed,models}
mkdir -p tests/{unit,integration}
mkdir -p docs/{architecture,setup}

# Initialize git
git init
git add .
git commit -m "Initial project structure"
```

### 1.2 Create Configuration Files

**requirements.txt:**
```
open-spiel==1.2
tensorflow==2.12.0
numpy==1.24.3
pandas==2.0.0
matplotlib==3.7.0
seaborn==0.12.0
pyyaml==6.0
scipy==1.10.0
tqdm==4.65.0
google-cloud-storage==2.10.0
```

**setup.py:**
```python
from setuptools import setup, find_packages

setup(
    name="multiagent-learning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "open-spiel>=1.2",
        "tensorflow>=2.12.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
    ],
)
```

**Makefile:**
```makefile
.PHONY: install test run-kuhn-nfsp run-kuhn-psro

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v

run-kuhn-nfsp:
	python experiments/runners/train_nfsp.py --game kuhn_poker --seed 0 --num_episodes 10000

run-kuhn-psro:
	python experiments/runners/train_psro.py --game kuhn_poker --seed 0 --num_iterations 5
```

### 1.3 Setup Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Part 2: Implementation Phases

### Phase 1: Core Utilities (Days 1-2)

**Use these Cursor prompts to generate code:**

1. **Prompt 1.1** → `src/networks/networks.py` (MLP class)
2. **Prompt 1.2** → `src/networks/buffers.py` (CircularBuffer, ReservoirBuffer)
3. **Prompt 1.3** → `src/games/game_wrapper.py` (GameWrapper for OpenSpiel)
4. **Prompt 1.4** → `src/games/state_encoder.py` (Encoders for 4 games)
5. **Prompt 1.5** → `src/utils/exploitability.py` (Exploitability metrics)

**Test Phase 1:**
```bash
python -c "from src.networks.networks import MLP; print('Networks OK')"
python -c "from src.games.game_wrapper import GameWrapper; game = GameWrapper('kuhn_poker'); print('GameWrapper OK')"
```

### Phase 2: NFSP Implementation (Days 3-5)

**Use these Cursor prompts:**

1. **Prompt 2.1** → `src/algorithms/nfsp.py` (NFSPAgent class)
2. **Prompt 2.2** → `experiments/runners/train_nfsp.py` (Training loop)
3. **Prompt 2.3** → `tests/integration/test_nfsp_kuhn.py` (Unit tests)

**Test Phase 2 - Milestone: NFSP works on Kuhn Poker**
```bash
pytest tests/integration/test_nfsp_kuhn.py -v

# Quick 1000-episode test
python experiments/runners/train_nfsp.py \
  --game kuhn_poker \
  --num_episodes 1000 \
  --seed 0 \
  --output_dir ./tmp/

# Should produce: tmp/nfsp_kuhn_poker_0.csv with exploitability decreasing
```

### Phase 3: PSRO Implementation (Days 6-8)

**Use these Cursor prompts:**

1. **Prompt 3.1** → `src/algorithms/psro.py` Part 1 (Framework structure)
2. **Prompt 3.2** → `src/algorithms/psro.py` Part 2 (BR training)
3. **Prompt 3.3** → `src/algorithms/psro.py` Part 3 (Meta-strategy solver)
4. **Prompt 3.4** → `experiments/runners/train_psro.py` (Training loop)
5. **Prompt 3.5** → `tests/integration/test_psro_kuhn.py` (Unit tests)

**Test Phase 3 - Milestone: PSRO works on Kuhn Poker**
```bash
pytest tests/integration/test_psro_kuhn.py -v

# Quick 3-iteration test
python experiments/runners/train_psro.py \
  --game kuhn_poker \
  --num_iterations 3 \
  --seed 0 \
  --output_dir ./tmp/
```

### Phase 4: Configuration & Orchestration (Day 9)

**Use these Cursor prompts:**

1. **Prompt 4.1** → `experiments/configs/{kuhn,leduc,liarsdice,darkhex}.yaml`
2. **Prompt 4.2** → `experiments/gcp/submit_job.py` (GCP job submission)

### Phase 5: Analysis & Plotting (Day 10)

**Use these Cursor prompts:**

1. **Prompt 5.1** → `analysis/stats/aggregate_results.py` (Results aggregation)
2. **Prompt 5.2** → `analysis/plots/plot_exploitability.py` (Visualization)

---

## Part 3: Running on GCP

### 3.1 GCP Setup

```bash
# 1. Create GCP project
gcloud config set project multiagent-learning

# 2. Create storage bucket
gsutil mb gs://multiagent-learning-results/

# 3. Create service account (in GCP Console)
# Then download JSON key file

# 4. Authenticate
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
```

### 3.2 Create GCP Instances

```bash
# Create 5 instances for parallel experiments
for i in 0 1 2 3 4; do
  gcloud compute instances create ml-runner-$i \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --machine-type=n1-standard-8 \
    --zone=us-central1-a \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-ssd \
    --preemptible
done
```

### 3.3 Setup Code on Instances

**Create startup-script.sh:**
```bash
#!/bin/bash
cd /home/ubuntu
git clone https://github.com/YOUR-USERNAME/multiagent-learning-project.git
cd multiagent-learning-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
echo "Setup complete"
```

**Or manually SSH and setup:**
```bash
gcloud compute ssh ml-runner-0 --zone=us-central1-a

# Inside the instance:
git clone https://github.com/YOUR-USERNAME/multiagent-learning-project.git
cd multiagent-learning-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 3.4 Run Experiments on GCP

**Submit NFSP jobs:**
```bash
for seed in 0 1 2 3 4; do
  for game in kuhn_poker leduc_poker liars_dice dark_hex; do
    gcloud compute ssh ml-runner-$seed --zone=us-central1-a << EOF
cd ~/multiagent-learning-project
source venv/bin/activate

python experiments/runners/train_nfsp.py \
  --game $game \
  --seed $seed \
  --output_dir ~/results/

# Upload results to GCS
gsutil cp ~/results/nfsp_${game}_${seed}.csv \
  gs://multiagent-learning-results/nfsp/${game}/seed-${seed}/
EOF
  done
done
```

**Submit PSRO jobs (after NFSP completes):**
```bash
# Similar to above, but with train_psro.py
```

### 3.5 Monitor & Download Results

```bash
# Check instance status
gcloud compute instances list

# Download results locally
mkdir -p data/raw
gsutil -m cp -r gs://multiagent-learning-results/* data/raw/

# Aggregate and plot
python analysis/stats/aggregate_results.py \
  --input_dir data/raw/ \
  --output_dir data/processed/

python analysis/plots/plot_exploitability.py \
  --input data/processed/ \
  --output analysis/plots/output/

# View results
open analysis/plots/output/*.pdf
```

### 3.6 Cleanup

```bash
# Delete instances
gcloud compute instances delete ml-runner-{0..4} \
  --zone=us-central1-a

# Delete old results from GCS
gsutil -m rm -r gs://multiagent-learning-results/
```

---

## Part 4: Expected Results & Analysis

### Hypotheses to Test

**H1 (Sample Efficiency):** NFSP better in large state spaces, PSRO better in small games
**H2 (Final Convergence):** Both reach similar final exploitability
**H3 (Computational Cost):** PSRO higher per-iteration cost, fewer total iterations
**H4 (Game Scaling):** NFSP scales better with tree size

### Success Criteria

| Game | Target ε | Max Episodes/Iterations | NFSP Time | PSRO Time |
|------|----------|------------------------|-----------|-----------|
| Kuhn Poker | 0.01 | 10⁶ / 20 | ~10 min | ~2 hours |
| Leduc Poker | 0.05 | 10⁷ / 20 | ~2 hours | ~10 hours |
| Liar's Dice | 0.1 | 10⁷ / 20 | ~3 hours | ~15 hours |
| Dark Hex | 0.1 | 10⁷ / 20 | ~4 hours | ~20 hours |

---

## Part 5: Key Cursor Prompts Reference

| Phase | Prompt # | File | Task |
|-------|----------|------|------|
| 1 | 1.1 | `networks.py` | MLP neural network |
| 1 | 1.2 | `buffers.py` | Replay buffers |
| 1 | 1.3 | `game_wrapper.py` | Game interface |
| 1 | 1.4 | `state_encoder.py` | State encodings |
| 1 | 1.5 | `exploitability.py` | Metrics |
| 2 | 2.1 | `nfsp.py` | NFSP agent |
| 2 | 2.2 | `train_nfsp.py` | NFSP training loop |
| 2 | 2.3 | `test_nfsp_kuhn.py` | NFSP tests |
| 3 | 3.1-3.5 | `psro.py` | PSRO framework |
| 3 | 3.4 | `train_psro.py` | PSRO training loop |
| 3 | 3.5 | `test_psro_kuhn.py` | PSRO tests |
| 4 | 4.1 | `*.yaml` | Config files |
| 4 | 4.2 | `submit_job.py` | GCP submission |
| 5 | 5.1 | `aggregate_results.py` | Result aggregation |
| 5 | 5.2 | `plot_exploitability.py` | Visualization |

---

## Timeline Summary

```
Week 1: Core utilities + NFSP implementation
  - Days 1-2: Repository setup + Phase 1 (core utilities)
  - Days 3-5: Phase 2 (NFSP implementation)
  - Milestone: NFSP converges on Kuhn Poker (ε < 0.1)

Week 2: PSRO implementation + Configuration
  - Days 6-8: Phase 3 (PSRO implementation)
  - Days 9-10: Phase 4-5 (configs, analysis setup)
  - Milestone: PSRO converges on Kuhn Poker, infrastructure ready

Week 3: Run experiments
  - Days 11-14: Submit jobs to GCP (all 4 games)
  - Days 15-18: Monitor execution, collect results
  - Days 19-21: Aggregate, plot, analyze
  - Milestone: All results collected and visualized

Week 4: Write findings
  - Days 22-28: Write paper/report, interpret results
  - Optional: Implement XDO if time permits

Total: ~28 days active work, ~280 GPU-hours compute
```

---

## Troubleshooting

### Local Development Issues

**Problem:** OpenSpiel import error
```bash
# Solution: reinstall from source
pip uninstall open-spiel
pip install open-spiel --no-binary open-spiel
```

**Problem:** TensorFlow GPU not detected
```bash
# Solution: verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, install GPU drivers
# nvidia-smi should show your GPU
```

**Problem:** NFSP exploitability stays constant or NaN
```bash
# Solution: check learning rates and buffer sizes in config
# Try: lr_rl = 0.01, lr_sl = 0.001, batch_size = 64
```

### GCP Issues

**Problem:** "Permission denied" when uploading to GCS
```bash
# Solution: grant correct IAM roles
gcloud projects add-iam-policy-binding multiagent-learning \
  --member=serviceAccount:YOUR-SA@PROJECT.iam.gserviceaccount.com \
  --role=roles/storage.admin
```

**Problem:** Instance runs out of disk space
```bash
# Solution: stream results to GCS in real-time
# Add to training script:
if step % 10000 == 0:
    subprocess.run(['gsutil', 'cp', '-r', 'results/', 'gs://bucket/'])
```

**Problem:** Preemptible instance stopped unexpectedly
```bash
# Solution: implement checkpointing
# See GCP_EXECUTION_GUIDE.md for details
```

---

## Resources

- **Full Documentation:** 
  - `SETUP_GUIDE.md` - Detailed repo structure
  - `CURSOR_PROMPTS_CHEATSHEET.md` - All 15 Cursor prompts
  - `GCP_EXECUTION_GUIDE.md` - Complete GCP guide

- **Reference Paper:** Heinrich & Silver (2016) - "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games"

- **OpenSpiel:** https://github.com/deepmind/open_spiel

- **Google Cloud:** https://cloud.google.com/compute

---

## Final Checklist

Before starting:
- [ ] Repository structured locally
- [ ] requirements.txt created
- [ ] Virtual environment activated
- [ ] Cursor IDE ready

During implementation:
- [ ] Phase 1 utilities working
- [ ] NFSP converges on Kuhn Poker
- [ ] PSRO converges on Kuhn Poker
- [ ] Configuration files created
- [ ] GCP infrastructure set up

Before GCP run:
- [ ] All tests pass locally
- [ ] Git repo pushed to GitHub
- [ ] GCP project created
- [ ] Storage buckets created
- [ ] Service account configured
- [ ] Startup script tested

After experiments:
- [ ] Results downloaded locally
- [ ] Results aggregated
- [ ] Plots generated
- [ ] GCP instances deleted
- [ ] Findings documented

---

**Good luck! 🚀**
