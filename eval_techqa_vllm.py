#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
import torch

from sklearn.model_selection import train_test_split

from datasets import DatasetDict, Dataset
import pandas as pd
from transformers import AutoTokenizer

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   

from vllm import LLM, SamplingParams

SYS_PROMPT = (
    "You are a careful assistant for extractive question answering. "
    "Answer using only the given context. If the answer is not present, reply exactly: 'unanswerable'."
)
USER_TEMPLATE = (
    "Answer the question strictly based on the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def apply_chat_template(tokenizer, question, context):
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYS_PROMPT}\n\n" + USER_TEMPLATE.format(context=context, question=question)

ANSWER_SPLIT_RE = re.compile(r"(?i)\banswer\s*:\s*", re.MULTILINE)

def _strip_trailing_punct(s: str) -> str:
    s = s.rstrip()
    s = re.sub(r'[\"”’)\]]+$', '', s)
    s = re.sub(r'[.!?]+$', '', s)
    return s.strip()

def extract_answer(generated_text: str) -> str:
    text = (generated_text or "").strip()
    text = _strip_trailing_punct(text)
    parts = ANSWER_SPLIT_RE.split(text)
    tail = parts[-1] if parts else text
    tail = tail.split("\n\n")[0].strip().split("\n")[0].strip()
    for prefix in ("The answer is", "Answer is", "It is", "It's", "This is"):
        if tail.lower().startswith(prefix.lower()):
            tail = tail[len(prefix):].strip(" :-.\"“”’'")
    tail = _strip_trailing_punct(tail)[:512].strip()
    low = tail.lower()
    if low in {"", "n/a", "not available", "cannot be determined", "unanswerable"}:
        return "unanswerable"
    if any(p in low for p in (
        "not present in the context",
        "cannot be answered from the context",
        "no answer",
        "unanswerable",
        "not enough information",
        "not provided in the context",
    )):
        return "unanswerable"
    return tail

def to_examples(ds):
    out = []
    for ex in ds:
        golds = ex.get("answers", {}).get("text", [])
        out.append({
            "qid": ex["id"],
            "question": ex["question"],
            "context": ex["context"],
            "golds": golds,
        })
    return out

def vllm_generate(llm, prompts, max_new_tokens, temperature, top_p, seed):
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["<|eot_id|>", "<|end_of_turn|>"],
        seed=seed,
    )
    outputs = llm.generate(prompts, sp)
    return [o.outputs[0].text.lstrip() if o.outputs else "" for o in outputs]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct-offline")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_input_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    
    # Quantization options
    parser.add_argument("--bnb8", action="store_true", help="Load model in 8-bit via bitsandbytes (vLLM).")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit via bitsandbytes (vLLM).")
    
    parser.add_argument("--bf16", action="store_true", help="(ignored) kept for CLI compatibility")
    parser.add_argument("--output_pred", default="predictions_techqa_llama3.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    print("[BOOT] starting run")
    torch.manual_seed(args.seed)


    qa_paths = {
        "train": {
            "qa": "/home/brachmat/phd/datasets/TechQA/training_and_dev/training_Q_A.json",
            "doc": "/home/brachmat/phd/datasets/TechQA/training_and_dev/training_dev_technotes.json"
        },
        "dev": {
            "qa": "/home/brachmat/phd/datasets/TechQA/training_and_dev/dev_Q_A.json",
            "doc": "/home/brachmat/phd/datasets/TechQA/training_and_dev/training_dev_technotes.json"
        },
        "validation": {
            "qa": "/home/brachmat/phd/datasets/TechQA/validation/validation_reference.json",
            "doc": "/home/brachmat/phd/datasets/TechQA/validation/validation_technotes.json"
        }
    }

    def safe_int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return -1

    def load_split(name, qa_path, doc_path):
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        with open(doc_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        records = []
        for q in qa_data:
            question_id = q.get("QUESTION_ID", "").strip()
            question_text = q.get("QUESTION_TEXT", "").strip()
            answer_text = q.get("ANSWER", "").strip()
            passage_id = q.get("DOCUMENT", "").strip()
            passage_entry = doc_data.get(passage_id, {})

            passage_title = passage_entry.get("title", "").strip()
            passage_text = passage_entry.get("text") or passage_entry.get("content", "")

            start_offset = safe_int(q.get("START_OFFSET"))
            end_offset = safe_int(q.get("END_OFFSET"))
            answerable = int(q.get("ANSWERABLE", "").strip().upper() == "Y")

            records.append({
                "split": name,
                "id": question_id,
                "context": passage_text,
                "title": passage_title,
                "question": question_text,
                "answer": answer_text,
                "answer_start": start_offset,
                "answer_end": end_offset,
                "answerable": answerable
            })
        return pd.DataFrame(records)

    df = pd.concat(
        [load_split(name, paths["qa"], paths["doc"]) for name, paths in qa_paths.items()],
        ignore_index=True
    )
    
    dev_df = df[df["split"] == "dev"]

    dev_train, dev_val = train_test_split(dev_df, test_size=0.3, random_state=42)

    dev_train = dev_train.copy()
    dev_val = dev_val.copy()
    dev_train["split"] = "train"
    dev_val["split"] = "validation"

    df = pd.concat([df[df["split"] != "dev"], dev_train, dev_val], ignore_index=True)

    # df = df[df['context']!= ""]
    df["split"] = df["split"].replace("dev", "validation")
    
    df = df[df['split'] == args.split]
    df_hf = df.copy()

    df_hf = df_hf[['id', 'title', 'context', 'question', 'answer', 'answer_start', 'answer_end']]

    df_hf["answers"] = df_hf.apply(
        lambda row: {
            "text": [row["answer"]] if row["answer"] else [],
            "answer_start": [row["answer_start"]] if row["answer_start"] != -1 else []
        },
        axis=1
    )

    df_hf = df_hf.drop(columns=["answer", "answer_start", "answer_end"])

    ds = Dataset.from_pandas(df_hf, preserve_index=False)
    
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"[Info] Limiting to {len(ds)} examples for a quick test.")
    else:
        print(f"[Info] Using full dataset: {len(ds)} examples")
    examples = to_examples(ds)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",                
        gpu_memory_utilization=0.75,
        max_model_len=args.max_input_tokens,
        tensor_parallel_size=1,
    )

    # Bitsandbytes quantization
    if args.load_in_4bit:
        llm_kwargs["quantization"] = "bitsandbytes"  # 4-bit (nf4) in vLLM
    elif args.bnb8:
        llm_kwargs["quantization"] = "bitsandbytes"  # vLLM will use 8-bit when not forcing 4-bit

    print("[BOOT] starting vLLM engine on CUDA_VISIBLE_DEVICES=1 …")
    llm = LLM(**llm_kwargs)

    # all_prompts = [apply_chat_template(tokenizer, ex["question"], ex["context"]) for ex in examples]
    kept_examples, all_prompts = [], []
    budget_for_prompt = args.max_input_tokens - args.max_new_tokens  # leave room for generation

    for ex in examples:
        prompt = apply_chat_template(tokenizer, ex["question"], ex["context"])
        n_tokens = count_tokens(tokenizer, prompt)

        if n_tokens <= budget_for_prompt:
            kept_examples.append(ex)
            all_prompts.append(prompt)
        else:
            print(
                f"[SKIP] {ex.get('qid','?')}: prompt tokens={n_tokens} > budget={budget_for_prompt}.",
                file=sys.stderr, flush=True
            )

    examples = kept_examples


    n = len(examples)
    bs = args.batch_size
    predictions, references = [], []

    print("[BOOT] starting inference…")
    with open(args.output_pred, "w", encoding="utf-8") as fout:
        for i in range(0, n, bs):
            batch = examples[i:i+bs]
            prompts = all_prompts[i:i+bs]

            decoded = vllm_generate(
                llm, prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )

            for ex, cont_text in zip(batch, decoded):
                pred = extract_answer(cont_text)
                golds = ex["golds"]
                gold_one = golds[0] if golds else ""

                fout.write(json.dumps({
                    "id": ex["qid"],
                    "context": ex["context"],
                    "question": ex["question"],
                    "predicted_answer": pred,
                    "gold_answer": gold_one
                }, ensure_ascii=False) + "\n")

                predictions.append({"id": ex["qid"], "prediction_text": pred, "no_answer_probability": 0.0})
                references.append({"id": ex["qid"], "answers": {"text": golds, "answer_start": [-1]*len(golds)}})

            done = min(i + bs, n)
            if (i // bs) % 10 == 0 or done == n:
                print(f"[Progress] {done}/{n}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
