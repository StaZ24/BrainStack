#!/bin/bash

# Get manually specified GPU ID (default to 0 if not set)
manual_gpu=${1:-0}
note=${2}

if [[ -z "$note" ]]; then
  echo "❌ Error: You must provide an experiment note as the second argument."
  echo "👉 Usage: bash run_all_subjects.sh [GPU_ID] [NOTE]"
  exit 1
fi

timestamp=$(date +"%Y-%m-%d_%H-%M")
snapshot_path="$snapshot_dir/main_${note}_${timestamp}.py"
cp main.py "$snapshot_path"
echo "📌 Saved snapshot: $snapshot_path"

base_dir="/projects/logs_ensemble_learning"
snapshot_dir="$base_dir/main_snapshots"
mkdir -p "$base_dir" "$snapshot_dir"

max_jobs=5
running_jobs=0

subjects=("S02" "S03" "S04" "S05" "S06" "S07" "S08" "S09" "S10")
declare -a all_logs

run_subject () {
  subject="$1"
  timestamp=$(date +"%Y-%m-%d_%H-%M")
  raw_log="$base_dir/logs_${subject}_${timestamp}.log"

  echo "▶️ [$(date +"%H:%M:%S")] Launching $subject on GPU $manual_gpu"
  echo "↪ Writing raw log to: $raw_log"

  CUDA_VISIBLE_DEVICES=$manual_gpu python main.py --note roi5 --subject "$subject" > "$raw_log" 2>&1

  test_acc=$(grep -i "Test Accuracy" "$raw_log" | tail -1 | grep -oE "[0-9]+\.[0-9]+")

  formatted_acc=$(echo "$test_acc" | sed 's/\./p/')
  final_log="$base_dir/logs_${subject}_acc${formatted_acc}_${timestamp}.log"

  mv "$raw_log" "$final_log"
  all_logs+=("$final_log")

  echo "✅ [$(date +"%H:%M:%S")] Finished $subject → Accuracy: $test_acc% → GPU $manual_gpu → $final_log"
}

for subject in "${subjects[@]}"; do
  run_subject "$subject" & 
  ((running_jobs+=1))

  if [[ $running_jobs -ge $max_jobs ]]; then
    wait -n
    ((running_jobs-=1))
    echo "📊 [$(date +"%H:%M:%S")] Waiting slot freed. Running jobs: $((running_jobs))"
  fi
done

wait

# Summary
echo -e "\n📦 All subjects completed. Logs:"
for log_path in "${all_logs[@]}"; do
  echo " - $log_path"
done