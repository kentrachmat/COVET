python eval_squadv2_vllm.py \
  --model /export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct-offline \
  --split validation \
  --batch_size  8 \
  --limit 0

python eval_squadv2_vllm.py \
  --model /home/brachmat/phd/models/Qwen2.5-7B-Instruct \
  --split validation \
  --batch_size  8 \
  --limit 0

python eval_squadv2_vllm.py \
  --model /export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct-offline \
  --split validation \
  --batch_size  8 \
  --bnb8 \
  --limit 10

python eval_squadv2_bert.py \
  --model /home/brachmat/phd/models/bert-base-uncased \
  --split validation \
  --batch_size  8 \
  --limit 0

python eval_squadv2_bert.py \
  --model /home/brachmat/phd/models/roberta-large \
  --split validation \
  --batch_size  8 \
  --limit 0



  python augmentation.py \
  --model /export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct-offline \
  --split train \
  --bnb8 \
  --limit 0