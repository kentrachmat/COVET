#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
import sys
import uuid
from typing import Any, Dict, List, Tuple

import pandas as pd

import torch
from datasets import load_dataset, Dataset, DatasetDict

from sklearn.model_selection import train_test_split


from transformers import AutoTokenizer

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from vllm import LLM, SamplingParams

# -----------------------
# Prompt templates
# -----------------------

JSON_SYS_PROMPT = (
    "You are a careful data augmentation engine. "
    "Always return ONLY valid, strict JSON. Do not add explanations, code fences, or extra text. "
    "If any field is not applicable, still include it with an empty array."
)

SEMANTIC_USER_TMPL = """You are given a context and an original question. Create 3 new question-short answer pairs that are semantically diverse: each question should ask about a different facet of the context, not just paraphrase the original.

Return the result strictly in JSON with the following structure:
{{
  "questions": ["...", "...", "..."],
  "answers":   ["...", "...", "..."]
}}

----
Context:
{context}

Question:
{question}
"""

SYNTACTIC_USER_TMPL = """You are given a context and a question. Rewrite the question so that it asks for the same information but with different sentence structures. Vary between active/passive voice, simple vs. complex sentences, and alternative grammatical constructions. Keep the meaning identical.

Generate 3 syntactically diverse alternatives and its type of structure in 1 word.
Return the result strictly in JSON with the following structure:
{{
  "alternatives": ["...", "...", "..."],
  "type": ["...", "...", "..."]
}}

----
Context:
{context}

Question:
{question}
"""

LEXICAL_USER_TMPL = """You are given a context and a question. Rewrite the question so that it asks for the same information but uses different vocabulary and phrasing. Keep the meaning identical, but avoid repeating the same words. Generate 3 diverse alternatives.

Return the result strictly in JSON with the following structure:
{{
  "alternatives": ["...", "...", "..."]
}}

----
Context:
{context}

Question:
{question}
"""

# -----------------------
# Chat templating
# -----------------------

def apply_chat_template(tokenizer, user_content: str) -> str:
    """Wrap in chat format if model supports it."""
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": JSON_SYS_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback plain text
    return f"{JSON_SYS_PROMPT}\n\n{user_content}\n"

# -----------------------
# JSON parsing helpers
# -----------------------

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_first_json(text: str) -> Dict[str, Any]:
    """
    Try to parse the first JSON object found in the text.
    Raise ValueError if not found or invalid.
    """
    if not text:
        raise ValueError("Empty generation.")
    m = JSON_BLOCK_RE.search(text)
    if not m:
        raise ValueError("No JSON object detected.")
    block = m.group(0).strip()
    try:
        return json.loads(block)
    except Exception as e:
        raise ValueError(f"Invalid JSON: {e}\nBlock: {block[:3000]}")

def ensure_list_len(d: Dict[str, Any], key: str, n: int) -> List[str]:
    """Get list at key, coerce to list of strings length n (truncate/pad with '')."""
    val = d.get(key, [])
    if not isinstance(val, list):
        val = [val]
    val = [str(x).strip() for x in val]
    if len(val) < n:
        val = val + [""] * (n - len(val))
    return val[:n]

# -----------------------
# Data access
# -----------------------

def to_examples(ds) -> List[Dict[str, Any]]:
    out = []
    for ex in ds:
        out.append({
            "qid": ex["id"],
            "question": ex["question"],
            "context": ex["context"],
        })
    return out

# -----------------------
# vLLM generation
# -----------------------

def vllm_generate(llm: LLM, prompts: List[str], max_new_tokens: int, temperature: float, top_p: float, seed: int) -> List[str]:
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["<|eot_id|>", "<|end_of_turn|>"],
        seed=seed,
    )
    outputs = llm.generate(prompts, sp)
    return [o.outputs[0].text.lstrip() if o.outputs else "" for o in outputs]

# -----------------------
# Build prompts for each type
# -----------------------

def build_prompts_for_row(tokenizer, context: str, question: str) -> Dict[str, str]:
    sem_user = SEMANTIC_USER_TMPL.format(context=context, question=question)
    syn_user = SYNTACTIC_USER_TMPL.format(context=context, question=question)
    lex_user = LEXICAL_USER_TMPL.format(context=context, question=question)
    return {
        "semantic": apply_chat_template(tokenizer, sem_user),
        "syntactic": apply_chat_template(tokenizer, syn_user),
        "lexical": apply_chat_template(tokenizer, lex_user),
    }

# -----------------------
# Flatteners to per-item rows
# -----------------------

def flatten_semantic(orig_id: str, orig_q: str, ctx: str, parsed: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    qs = ensure_list_len(parsed, "questions", 3)
    ans = ensure_list_len(parsed, "answers", 3)
    rows = []
    for i, (q, a) in enumerate(zip(qs, ans)):
        rows.append({
            "aug_id": f"{orig_id}::SEM::{i}",
            "orig_id": orig_id,
            "type": "SEMANTIC",
            "alt_index": i,
            "question": q,
            "answer": a,
            "orig_question": orig_q,
            "context": ctx,
        })
    return rows, {"questions": qs, "answers": ans}

def flatten_syntactic(orig_id: str, orig_q: str, ctx: str, parsed: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    alts = ensure_list_len(parsed, "alternatives", 3)
    typs = ensure_list_len(parsed, "type", 3)
    rows = []
    for i, (q, t) in enumerate(zip(alts, typs)):
        rows.append({
            "aug_id": f"{orig_id}::SYN::{i}",
            "orig_id": orig_id,
            "type": "SYNTACTIC",
            "alt_index": i,
            "question": q,
            "answer": "",        # no answer needed here
            "structure_type": t, # extra field
            "orig_question": orig_q,
            "context": ctx,
        })
    return rows, {"alternatives": alts, "type": typs}

def flatten_lexical(orig_id: str, orig_q: str, ctx: str, parsed: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    alts = ensure_list_len(parsed, "alternatives", 3)
    rows = []
    for i, q in enumerate(alts):
        rows.append({
            "aug_id": f"{orig_id}::LEX::{i}",
            "orig_id": orig_id,
            "type": "LEXICAL",
            "alt_index": i,
            "question": q,
            "answer": "",        # no answer needed here
            "orig_question": orig_q,
            "context": ctx,
        })
    return rows, {"alternatives": alts}




# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct-offline")
    parser.add_argument("--split", default="train")   
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_input_tokens", type=int, default=60000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=5)   

    # Quantization knobs
    parser.add_argument("--bnb8", action="store_true", help="bitsandbytes 8-bit in vLLM")
    parser.add_argument("--load_in_4bit", action="store_true", help="bitsandbytes 4-bit (nf4) in vLLM")

    # Outputs
    parser.add_argument("--out_jsonl", default="augmented_techqa.jsonl")
    parser.add_argument("--out_csv", default="augmented_techqa.csv")

    args = parser.parse_args()

    print("[BOOT] starting augmentation run")
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

    df = df[df['context']!= ""]
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

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=args.max_input_tokens,
        tensor_parallel_size=1,
    )
    if args.load_in_4bit:
        llm_kwargs["quantization"] = "bitsandbytes"  # vLLM will do 4-bit (nf4)
    elif args.bnb8:
        llm_kwargs["quantization"] = "bitsandbytes"  # 8-bit

    print(f"[BOOT] starting vLLM engine on CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')} …")
    llm = LLM(**llm_kwargs)

    n = len(examples)
    bs = args.batch_size

    # Prepare writers
    csv_fieldnames = [
        "aug_id","orig_id","type","alt_index","question","answer","structure_type","orig_question","context"
    ]
    with open(args.out_jsonl, "w", encoding="utf-8") as f_jsonl, \
         open(args.out_csv, "w", encoding="utf-8", newline="") as f_csv:

        csv_writer = csv.DictWriter(f_csv, fieldnames=csv_fieldnames)
        csv_writer.writeheader()

        for i in range(0, n, bs):
            batch = examples[i:i+bs]

            # Build three prompts per item
            sem_prompts, syn_prompts, lex_prompts = [], [], []
            meta = []  # keep (orig_id, orig_q, ctx) aligned to each prompt triple
            for ex in batch:
                ctx = ex["context"]
                q = ex["question"]
                pid = ex["qid"]
                prompts = build_prompts_for_row(tokenizer, ctx, q)
                sem_prompts.append(prompts["semantic"])
                syn_prompts.append(prompts["syntactic"])
                lex_prompts.append(prompts["lexical"])
                meta.append((pid, q, ctx))

            # Generate for each prompt type (separately to keep alignment & memory)
            decoded_sem = vllm_generate(
                llm, sem_prompts, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p, seed=args.seed
            )
            decoded_syn = vllm_generate(
                llm, syn_prompts, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p, seed=args.seed
            )
            decoded_lex = vllm_generate(
                llm, lex_prompts, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p, seed=args.seed
            )

            # Parse & write
            for (orig_id, orig_q, ctx), sem_txt, syn_txt, lex_txt in zip(meta, decoded_sem, decoded_syn, decoded_lex):
                # SEMANTIC
                try:
                    sem_parsed = extract_first_json(sem_txt)
                    rows, kept = flatten_semantic(orig_id, orig_q, ctx, sem_parsed)
                    for r in rows:
                        f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")
                        csv_writer.writerow({k: r.get(k, "") for k in csv_fieldnames})
                except Exception:
                    stub_id = f"{orig_id}::SEM::ERR"
                    err_row = {
                        "aug_id": stub_id, "orig_id": orig_id, "type": "SEMANTIC",
                        "alt_index": -1, "question": "", "answer": "",
                        "structure_type": "", "orig_question": orig_q, "context": ctx
                    }
                    f_jsonl.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                    csv_writer.writerow({k: err_row.get(k, "") for k in csv_fieldnames})

                # SYNTACTIC
                try:
                    syn_parsed = extract_first_json(syn_txt)
                    rows, kept = flatten_syntactic(orig_id, orig_q, ctx, syn_parsed)
                    for r in rows:
                        f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")
                        csv_writer.writerow({k: r.get(k, "") for k in csv_fieldnames})
                except Exception:
                    stub_id = f"{orig_id}::SYN::ERR"
                    err_row = {
                        "aug_id": stub_id, "orig_id": orig_id, "type": "SYNTACTIC",
                        "alt_index": -1, "question": "", "answer": "",
                        "structure_type": "", "orig_question": orig_q, "context": ctx
                    }
                    f_jsonl.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                    csv_writer.writerow({k: err_row.get(k, "") for k in csv_fieldnames})

                # LEXICAL
                try:
                    lex_parsed = extract_first_json(lex_txt)
                    rows, kept = flatten_lexical(orig_id, orig_q, ctx, lex_parsed)
                    for r in rows:
                        f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")
                        csv_writer.writerow({k: r.get(k, "") for k in csv_fieldnames})
                except Exception:
                    stub_id = f"{orig_id}::LEX::ERR"
                    err_row = {
                        "aug_id": stub_id, "orig_id": orig_id, "type": "LEXICAL",
                        "alt_index": -1, "question": "", "answer": "",
                        "structure_type": "", "orig_question": orig_q, "context": ctx
                    }
                    f_jsonl.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                    csv_writer.writerow({k: err_row.get(k, "") for k in csv_fieldnames})

            done = min(i + bs, n)
            if (i // bs) % 5 == 0 or done == n:
                print(f"[Progress] {done}/{n}", file=sys.stderr, flush=True)

    print("✅ Augmentation finished.")
    print(f"JSONL: {os.path.abspath(args.out_jsonl)}")
    print(f"CSV:   {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()