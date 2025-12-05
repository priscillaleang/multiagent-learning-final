================================================================================
GOOGLE CLOUD PLATFORM (GCP) EXECUTION GUIDE FOR EXPERIMENTS
================================================================================

Complete guide to running your NFSP & PSRO experiments on GCP Compute Engine.

================================================================================
SETUP: GCP Project & Authentication
================================================================================

Step 1: Create GCP Project
1. Go to console.cloud.google.com
2. Create new project: "multiagent-learning"
3. Enable APIs:
   - Compute Engine API
   - Cloud Storage API
   - Cloud Logging API
4. Create service account for authentication

Step 2: Authentication
```bash
# Download service account JSON from GCP Console
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Authenticate gcloud
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
gcloud config set project multiagent-learning
```

Step 3: Create Storage Bucket
```bash
gsutil mb gs://multiagent-learning-results/
gsutil mb gs://multiagent-learning-code/
```

================================================================================
GCP INSTANCE SETUP
================================================================================

Create Compute Engine instances for parallel experimentation:

**Via gcloud CLI:**
```bash
gcloud compute instances create ml-runner-kuhn \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --machine-type=n1-standard-8 \
  --zone=us-central1-a \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd \
  --preemptible  # Cheaper but can be interrupted
```

================================================================================
SETUP: Code on Instances
================================================================================

SSH into an instance:
```bash
gcloud compute ssh ml-runner-kuhn --zone=us-central1-a
```

Setup code and environment:
```bash
cd /home/$(whoami)

# Clone repo
git clone https://github.com/YOUR-USERNAME/multiagent-learning-project.git
cd multiagent-learning-project

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install deps
pip install -r requirements.txt
pip install -e .

# Test import
python -c "import pyspiel; print('OpenSpiel OK')"
```

Create startup script to automate this:
**startup-script.sh:**
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

Use with:
```bash
gcloud compute instances create ml-runner-kuhn \
  ... \
  --metadata-from-file startup-script=startup-script.sh
```

================================================================================
EXECUTION: Running Experiments on Instances
================================================================================

Method 1: SSH and Run Directly
```bash
# SSH into instance
gcloud compute ssh ml-runner-kuhn --zone=us-central1-a

# Inside instance:
cd ~/multiagent-learning-project
source venv/bin/activate

# Run NFSP on Kuhn Poker
python experiments/runners/train_nfsp.py \
  --game kuhn_poker \
  --num_episodes 1000000 \
  --seed 0 \
  --output_dir ~/results/ \
  --eval_freq 1000

# Monitor progress (leave running)
```

Method 2: Background Job with tmux (Better)
```bash
gcloud compute ssh ml-runner-kuhn --zone=us-central1-a << 'EOF'
cd ~/multiagent-learning-project
source venv/bin/activate

# Start tmux session
tmux new-session -d -s nfsp "python experiments/runners/train_nfsp.py --game kuhn_poker --seed 0 --output_dir ~/results/"

# Attach to monitor
# tmux attach-session -t nfsp
EOF
```

Method 3: Upload Results to GCS
```bash
# Still inside instance after training completes:
gsutil -m cp ~/results/*.csv gs://multiagent-learning-results/nfsp/kuhn/seed-0/
```

Method 4: Download Results Locally
```bash
# On your local machine:
gsutil -m cp gs://multiagent-learning-results/nfsp/kuhn/seed-*/*.csv ./data/raw/
```

================================================================================
ORCHESTRATION: Parallel Experiments Script
================================================================================

**Run on local machine to submit jobs to all instances:**

**scripts/run_experiments_gcp.sh:**
```bash
#!/bin/bash

# Configuration
ZONES="us-central1-a us-central1-b"
GAMES="kuhn_poker leduc_poker liars_dice dark_hex"
ALGORITHMS="nfsp psro"
SEEDS="0 1 2 3 4"
OUTPUT_BUCKET="gs://multiagent-learning-results"

# Create instances if needed
for game in $GAMES; do
  for i in 0 1 2 3 4; do
    INSTANCE_NAME="ml-${game}-${i}"
    echo "Creating instance: $INSTANCE_NAME"
    gcloud compute instances create $INSTANCE_NAME \
      --image-family=ubuntu-2004-lts \
      --image-project=ubuntu-os-cloud \
      --machine-type=n1-standard-8 \
      --zone=us-central1-a \
      --preemptible \
      --metadata-from-file startup-script=startup-script.sh || true
  done
done

# Submit training jobs
counter=0
for algo in $ALGORITHMS; do
  for game in $GAMES; do
    for seed in $SEEDS; do
      ZONE="us-central1-a"
      INSTANCE_NAME="ml-${game}-${seed}"
      
      echo "Submitting: $algo on $game seed $seed"
      
      gcloud compute ssh $INSTANCE_NAME --zone=$ZONE << EOF
cd ~/multiagent-learning-project
source venv/bin/activate

python experiments/runners/train_${algo}.py \
  --game $game \
  --seed $seed \
  --output_dir ~/results/ \
  | tee ~/logs/${algo}_${game}_${seed}.log

# Upload results to GCS
gsutil cp ~/results/${algo}_${game}_${seed}.csv \
  gs://multiagent-learning-results/${algo}/${game}/seed-${seed}/

echo "Completed: $algo on $game seed $seed"
EOF
      
      counter=$((counter + 1))
    done
  done
done

echo "Submitted $counter training jobs"
```

Run with:
```bash
bash scripts/run_experiments_gcp.sh
```

================================================================================
MONITORING: Track Running Jobs
================================================================================

**Check Instance Status:**
```bash
# List all instances
gcloud compute instances list

# Get serial port output (logs)
gcloud compute instances get-serial-port-output ml-runner-kuhn \
  --zone=us-central1-a

# SSH and monitor in real-time
gcloud compute ssh ml-runner-kuhn --zone=us-central1-a

# Inside instance: check running process
ps aux | grep python

# Kill a job if needed
pkill -f "train_nfsp"
```

**Monitor Disk Usage:**
```bash
gcloud compute ssh ml-runner-kuhn --zone=us-central1-a
df -h
du -sh ~/results/
```

================================================================================
COST OPTIMIZATION
================================================================================

**Use Preemptible Instances (saves ~70%):**
```bash
--preemptible  # Add this flag to gcloud compute instances create
```
Risk: Can be stopped with 30-second notice. For NFSP/PSRO:
- Implement checkpointing every 10 mins
- Resume from checkpoint on restart

**Use Committed Discounts:**
- 1-year commitment: saves ~25%
- 3-year commitment: saves ~50%

**Estimate costs:**
- n1-standard-8: ~$0.38/hour (on-demand)
- With T4 GPU: +$0.35/hour
- Preemptible discount: -~70%
- Monthly estimate: 730 hours * $0.38 = ~$277 (compute only)
- With GPU: ~$534

**For 5 games × 2 algorithms × 5 seeds:**
- Total compute: ~200 GPU-hours
- Cost at on-demand rates: ~$3,500
- Cost with preemptible + commitment: ~$400

================================================================================
AGGREGATING RESULTS
================================================================================

After experiments complete, aggregate and plot:

**1. Download all results:**
```bash
mkdir -p data/raw
gsutil -m cp -r gs://multiagent-learning-results/* data/raw/

# Check structure
find data/raw -name "*.csv" | head -20
```

**2. Aggregate locally:**
```bash
python analysis/stats/aggregate_results.py \
  --input_dir data/raw/ \
  --output_dir data/processed/

# Check output
head -5 data/processed/summary_results.csv
```

**3. Plot results:**
```bash
python analysis/plots/plot_exploitability.py \
  --input data/processed/ \
  --output analysis/plots/output/

# View plots
open analysis/plots/output/*.pdf
```

================================================================================
CHECKPOINT & RESUME
================================================================================

To handle instance interruptions, implement checkpointing:

**In train_nfsp.py:**
```python
def save_checkpoint(agents, episode, output_dir):
    checkpoint = {
        'episode': episode,
        'agent1_weights': agents[0].Q_network.weights,
        'agent2_weights': agents[1].Q_network.weights,
        'timestamp': time.time()
    }
    tf.saved_model.save(agents[0].Q_network, f"{output_dir}/ckpt_agent1_ep{episode}")
    tf.saved_model.save(agents[1].Q_network, f"{output_dir}/ckpt_agent2_ep{episode}")
    pickle.dump(checkpoint, open(f"{output_dir}/checkpoint_ep{episode}.pkl", 'wb'))

def load_checkpoint(checkpoint_dir):
    # Find latest checkpoint
    # Load weights back
    pass

# In main loop:
if episode % 50000 == 0:  # Checkpoint every 50k episodes
    save_checkpoint(agents, episode, output_dir)
```

**Auto-restart on preemption:**
```bash
# Add to startup script:
#!/bin/bash
...
# Resume from checkpoint if exists
latest_ckpt=$(ls -t ~/results/checkpoint*.pkl 2>/dev/null | head -1)
if [ -n "$latest_ckpt" ]; then
  echo "Resuming from checkpoint: $latest_ckpt"
  python experiments/runners/train_nfsp.py --resume $latest_ckpt
else
  echo "Starting fresh"
  python experiments/runners/train_nfsp.py
fi
```

================================================================================
CLEANUP & COST CONTROL
================================================================================

**Delete instances after experiments:**
```bash
# Delete single instance
gcloud compute instances delete ml-runner-kuhn --zone=us-central1-a

# Delete all instances matching pattern
gcloud compute instances delete $(gcloud compute instances list \
  --filter="name~'ml-runner'" --format='value(name)') --zone=us-central1-a
```

**Delete old storage:**
```bash
# Delete old results
gsutil -m rm -r gs://multiagent-learning-results/old-run-*/

# Set lifecycle policy to auto-delete old files
gsutil lifecycle set - gs://multiagent-learning-results/ <<< '{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}'
```

================================================================================
TROUBLESHOOTING
================================================================================

**Problem: "Permission denied" errors**
Solution: Check service account has necessary roles:
```bash
# Grant roles
gcloud projects add-iam-policy-binding multiagent-learning \
  --member=serviceAccount:YOUR-SA@PROJECT.iam.gserviceaccount.com \
  --role=roles/compute.admin
```

**Problem: Instance runs out of disk space**
Solution: Increase boot disk or stream results to GCS:
```bash
# Real-time upload
while true; do
  gsutil -m cp ~/results/*.csv gs://multiagent-learning-results/ 2>/dev/null
  rm ~/results/*.csv
  sleep 3600
done
```

**Problem: NFSP converges to NaN**
Solution: Check learning rates, buffer sizes in config:
```yaml
nfsp:
  lr_rl: 0.01    # Try smaller
  lr_sl: 0.001   # Try smaller
  batch_size: 64 # Try smaller
```

**Problem: Slow disk I/O**
Solution: Use SSD instead of standard disk:
```bash
--boot-disk-type=pd-ssd
```

================================================================================
FINAL CHECKLIST
================================================================================

Before running experiments:
- [ ] GCP project created and authenticated
- [ ] Storage bucket created (gs://multiagent-learning-results/)
- [ ] Service account with correct permissions
- [ ] Startup script tested on one instance
- [ ] Code repo pushed to GitHub
- [ ] requirements.txt updated and tested
- [ ] tests/ pass locally (pytest tests/)
- [ ] Sample training run completes on local machine
- [ ] Output CSV format verified
- [ ] Checkpointing implemented and tested

For each experiment batch:
- [ ] Create instances with gcloud CLI
- [ ] Monitor first job (1-2 hours) to verify it works
- [ ] Scale to 5 instances once verified
- [ ] Monitor costs in GCP Console
- [ ] Download results daily to local backup
- [ ] Delete instances immediately when done

================================================================================
