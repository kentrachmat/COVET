python eval_squadv2_vllm.py \
  --model /home/brachmat/phd/models/Qwen2.5-7B-Instruct \
  --split validation \
  --batch_size  8 \
  --limit 0


python eval_techqa_vllm.py \
  --model /home/brachmat/phd/models/Qwen2.5-7B-Instruct \
  --split validation \
  --batch_size  8 \
  --bnb8 \
  --limit 10

  python augmentation.py \
  --model /export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct-offline \
  --split train \
  --bnb8 \
  --limit 0

python augmentation_techqa.py \
  --model /export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct-offline \
  --split train \
  --bnb8 \
  --limit 0