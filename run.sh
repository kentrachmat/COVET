#!/bin/bash

declare -A models
# models["qwen"]="/home/brachmat/phd/models/Qwen2.5-7B-Instruct"
# models["llama"]="/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct-offline"
models["gemma"]="/export/home/cache/hub/unsloth-gemma-3-12b-it-offline"

for name in "${!models[@]}"; do
    model="${models[$name]}"
    echo ">>> Running evaluation for $name"

    # Build args safely
    args=(
      --model "$model"
      --split validation
      --batch_size 8
      --bnb8
      --limit 3
      --output_pred "predictions_techqa_${name}.jsonl"
    )

    if [[ "$name" != "gemma" ]]; then
      args+=(--vllm 1)
    fi

    python eval_techqa_vllm.py "${args[@]}"
done
