#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from typing import List, Dict

import pandas as pd
from openai import OpenAI

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

ANSWER_SPLIT_RE = re.compile(r"(?i)\banswer\s*:\s*", re.MULTILINE)

 
def safe_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return -1

def load_split(name: str, qa_path: str, doc_path: str) -> pd.DataFrame:
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

        passage_title = (passage_entry.get("title", "") or "").strip()
        passage_text = passage_entry.get("text") or passage_entry.get("content", "") or ""

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

def build_dataframe() -> pd.DataFrame:
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

    df = pd.concat(
        [load_split(name, paths["qa"], paths["doc"]) for name, paths in qa_paths.items()],
        ignore_index=True
    )

    df = df[df["answer"] != ""]

    dev_df = df[df["split"] == "dev"].sample(frac=1.0, random_state=42)
    cut = int(len(dev_df) * 0.7)
    dev_train = dev_df.iloc[:cut].copy()
    dev_val   = dev_df.iloc[cut:].copy()
    dev_train["split"] = "train"
    dev_val["split"]   = "validation"

    df = pd.concat([df[df["split"] != "dev"], dev_train, dev_val], ignore_index=True)

    # normalize any remaining "dev" label just in case
    df["split"] = df["split"].replace("dev", "validation")

    return df

def to_examples(df: pd.DataFrame) -> List[Dict]:
    examples = []
    for _, row in df.iterrows():
        gold = row["answer"] or ""
        examples.append({
            "qid": row["id"],
            "question": row["question"],
            "context": row["context"],
            "golds": [gold] if gold else [],
        })
    return examples

# ----------------------------
# Chat Completions call (Azure/OpenAI-compatible)
# ----------------------------
def ask_chat(client: OpenAI, deployment_name: str, question: str, context: str,
             temperature: float, top_p: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=deployment_name,  # for Azure-style, this is the *deployment* name
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(context=context, question=question)},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["<|eot_id|>", "<|end_of_turn|>"]
    )
    return (resp.choices[0].message.content or "").strip()

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    # Data selection
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output_pred", default="predictions_techqa_chat.jsonl")
    # API wiring (matches your snippet)
    parser.add_argument("--endpoint", default=os.getenv("OPENAI_BASE_URL", "https://ailab-kent-cifre.openai.azure.com/openai/v1/"))
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--deployment", default=os.getenv("OPENAI_DEPLOYMENT", "gpt-4o"))
    # Generation knobs
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] Provide --api_key or set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    print("[BOOT] loading TechQA …")
    df = build_dataframe()
    df = df[df["split"] == args.split].copy()

    if args.limit and args.limit > 0:
        df = df.head(args.limit)
        print(f"[Info] Limiting to {len(df)} examples for a quick test.")
    else:
        print(f"[Info] Using full dataset: {len(df)} examples")

    examples = to_examples(df)

    print("[BOOT] initializing client …")
    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    print("[BOOT] running inference via Chat Completions …")
    with open(args.output_pred, "w", encoding="utf-8") as fout:
        for idx, ex in enumerate(examples, 1):
            try:
                raw = ask_chat(
                    client,
                    deployment_name=args.deployment,
                    question=ex["question"],
                    context=ex["context"],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
            except Exception as e:
                print(f"[WARN] API error on {ex['qid']}: {e}", file=sys.stderr)
                raw = ""

            pred = extract_answer(raw)
            golds = ex["golds"]
            gold_one = golds[0] if golds else ""

            fout.write(json.dumps({
                "id": ex["qid"],
                "context": ex["context"],
                "question": ex["question"],
                "predicted_answer": pred,
                "gold_answer": gold_one
            }, ensure_ascii=False) + "\n")

            if idx % 10 == 0 or idx == len(examples):
                print(f"[Progress] {idx}/{len(examples)}", file=sys.stderr, flush=True)

    print(f"[DONE] wrote {args.output_pred}")

if __name__ == "__main__":
    main()