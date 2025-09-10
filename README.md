python run_squad_vllm.py \
  --model /export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct-offline \
  --split validation \
  --batch_size  8 \
  --limit 0


python run_squad_vllm.py \
  --model /home/brachmat/phd/models/Qwen2.5-7B-Instruct \
  --split validation \
  --batch_size  8 \
  --limit 0