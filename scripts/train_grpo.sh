#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
: "${DATA_ROOT:?Set DATA_ROOT to the source dataset directory}"
: "${SAM2_CHECKPOINT:?Set SAM2_CHECKPOINT to the SAM 2.1 checkpoint file}"
export DATA_ROOT SAM2_CHECKPOINT
export PYTHONPATH="$repo:$repo/site-packages/easyr1${PYTHONPATH:+:$PYTHONPATH}"
policy=${MODEL_PATH:-$repo/runs/sft/policy}
prepared=${PREPARED:-$repo/runs/prepared}
for file in "$prepared/train.jsonl" "$prepared/val.jsonl" "$SAM2_CHECKPOINT"; do
    [[ -s "$file" ]] || { echo "Missing or empty: $file" >&2; exit 1; }
done
python -m verl.trainer.main \
    config=site-packages/easyr1/examples/config.yaml \
    "data.train_files=$prepared/train.jsonl" "data.val_files=$prepared/val.jsonl" \
    "data.image_dir=$DATA_ROOT" data.format_prompt=null \
    data.max_prompt_length=4096 data.max_response_length=1536 data.min_pixels=3136 data.max_pixels=200704 \
    data.rollout_batch_size=8 data.val_batch_size=4 worker.actor.global_batch_size=8 \
    "worker.actor.model.model_path=$policy" worker.rollout.n=4 worker.rollout.tensor_parallel_size=1 \
    worker.rollout.neseg_max_calls=3 worker.rollout.limit_images=8 worker.rollout.gpu_memory_utilization=0.45 \
    worker.reward.reward_function=neseg/reward.py:compute_score \
    "worker.reward.reward_function_kwargs.data_root=$DATA_ROOT" \
    algorithm.adv_estimator=grpo trainer.project_name=neseg trainer.experiment_name=qwen25vl_grpo \
    trainer.val_generations_to_log=0 'trainer.logger=[console]' "trainer.n_gpus_per_node=${GPUS:-4}" "$@"
