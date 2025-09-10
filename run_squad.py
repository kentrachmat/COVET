#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate  # official SQuAD v2 metric

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

def apply_chat_template(tokenizer, question, context):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYS_PROMPT}\n\n" + USER_TEMPLATE.format(context=context, question=question)

ANSWER_SPLIT_RE = re.compile(r"(?i)\banswer:\s*", re.MULTILINE)

def extract_answer(generated_text):
    text = generated_text.strip()
    parts = ANSWER_SPLIT_RE.split(text)
    tail = parts[-1] if parts else text
    tail = tail.split("\n\n")[0].strip()
    for prefix in ["The answer is", "Answer is", "It is", "It's", "This is"]:
        if tail.lower().startswith(prefix.lower()):
            tail = tail[len(prefix):].strip(" :-.")
    tail = tail[:512].strip()
    low = tail.lower()
    if low in {"", "n/a", "not available", "cannot be determined", "unanswerable"}:
        return "unanswerable"
    if any(phrase in low for phrase in [
        "not present in the context",
        "cannot be answered from the context",
        "no answer",
        "unanswerable",
        "not enough information",
        "not provided in the context",
    ]):
        return "unanswerable"
    return tail

def to_examples(ds):
    out = []
    for ex in ds:
        golds = ex["answers"]["text"] if ex.get("answers") else []
        out.append({
            "qid": ex["id"],
            "question": ex["question"],
            "context": ex["context"],
            "golds": golds,
        })
    return out

def generate_batched(
    model, tokenizer, prompts, max_input_tokens, max_new_tokens,
    temperature, top_p, do_sample, eos_token_id,
):
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    input_lens = attention_mask.sum(dim=1).tolist()

    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            do_sample=do_sample, eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    outs = []
    for i in range(gen.size(0)):
        cont_ids = gen[i, input_lens[i]:]
        text = tokenizer.decode(cont_ids, skip_special_tokens=True)
        outs.append(text)
    return outs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_input_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output_pred", default="predictions_squad_llama3.jsonl")
    parser.add_argument("--output_eval", default="eval_squad_llama3.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    print("[BOOT] starting run")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ds = load_dataset("rajpurkar/squad_v2", split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"[Info] Limiting to {len(ds)} examples for a quick test.")
    else:
        print(f"[Info] Using full dataset: {len(ds)} examples")
    examples = to_examples(ds)

    tok_kwargs = {}
    tok_kwargs["padding_side"] = "left"
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(device_map="auto")
    if args.load_in_4bit:
        model_kwargs.update(
            dict(load_in_4bit=True,
                 bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16)
        )
    else:
        model_kwargs.update(dict(torch_dtype=torch.bfloat16 if args.bf16 else torch.float16))

    print("[BOOT] loading model…")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    eos_token_id = tokenizer.eos_token_id

    n = len(examples)
    bs = args.batch_size
    predictions = []
    references = []

    print("[BOOT] starting inference…")
    with open(args.output_pred, "w", encoding="utf-8") as fout:
        for i in range(0, n, bs):
            batch = examples[i:i+bs]
            prompts = [apply_chat_template(tokenizer, ex["question"], ex["context"]) for ex in batch]

            decoded = generate_batched(
                model, tokenizer, prompts,
                args.max_input_tokens, args.max_new_tokens,
                args.temperature, args.top_p, args.sample, eos_token_id
            )

            for ex, cont_text in zip(batch, decoded):
                pred = extract_answer(cont_text)
                golds = ex["golds"]
                gold_one = golds[0] if golds else ""

                # compact JSONL for you
                fout.write(json.dumps({
                    "id": ex["qid"],
                    "question": ex["question"],
                    "predicted_answer": pred,
                    "gold_answer": gold_one
                }, ensure_ascii=False) + "\n")

                predictions.append({
                    "id": ex["qid"],
                    "prediction_text": pred,
                    "no_answer_probability": 0.0   
                })
                references.append({
                    "id": ex["qid"],
                    "answers": {
                        "text": golds,
                        "answer_start": [-1] * len(golds)
                    }
                })

            done = min(i + bs, n)
            if (i // bs) % 10 == 0 or done == n:
                print(f"[Progress] {done}/{n}")

    # Evaluate (official)
    metric = evaluate.load("squad_v2")
    scores = metric.compute(predictions=predictions, references=references)

    summary = {
        "model": args.model,
        "split": args.split,
        "num_examples": scores.get("total", len(predictions)),
        "EM": round(scores.get("exact", 0.0), 3),
        "F1": round(scores.get("f1", 0.0), 3),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "sample": args.sample,
        "load_in_4bit": args.load_in_4bit,
        "bf16": args.bf16,
        "seed": args.seed,
        "predictions_file": os.path.abspath(args.output_pred),
        "has_answer_em": scores.get("HasAns_exact"),
        "has_answer_f1": scores.get("HasAns_f1"),
        "no_answer_em": scores.get("NoAns_exact"),
        "no_answer_f1": scores.get("NoAns_f1"),
    }
    with open(args.output_eval, "w") as fjs:
        json.dump(summary, fjs, indent=2)

    print("✅ Done.")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
