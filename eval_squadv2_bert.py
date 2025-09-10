#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


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


def load_qa_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tokenizer, model, device


def best_span_from_logits(start_logits: np.ndarray,
                          end_logits: np.ndarray,
                          max_answer_len: int = 30) -> Tuple[int, int, float]:
    """
    Constrained best (s,e) with e>=s and length<=max_answer_len.
    Returns (start_idx, end_idx, score=start_logits[s]+end_logits[e]).
    """
    best_s, best_e, best_score = 0, 0, -1e9
    # Consider top-k candidates for efficiency
    s_cands = np.argsort(start_logits)[-50:]
    for s in s_cands:
        e_min = s
        e_max = min(s + max_answer_len - 1, len(end_logits) - 1)
        e_slice = end_logits[e_min:e_max + 1]
        e_rel = int(e_slice.argmax())
        e = e_min + e_rel
        if e >= s:
            score = float(start_logits[s] + end_logits[e])
            if score > best_score:
                best_s, best_e, best_score = int(s), int(e), score
    return best_s, best_e, best_score


def run_qa(tokenizer, model, device, examples: List[Dict[str, Any]],
           qa_max_length: int, qa_doc_stride: int, qa_max_answer_len: int,
           na_threshold: float, batch_size: int) -> List[str]:
    tokenizer.padding_side = "right"

    answers: List[str] = []
    n = len(examples)

    print("[BOOT] starting QA (BERT/RoBERTa) inference…")
    for i in range(0, n, batch_size):
        batch = examples[i:i + batch_size]
        questions = [b["question"] for b in batch]
        contexts = [b["context"] for b in batch]

        enc = tokenizer(
            questions,
            contexts,
            return_tensors="pt",
            truncation="only_second",
            max_length=qa_max_length,
            stride=qa_doc_stride,
            padding=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )

        # Build inputs (token_type_ids may be absent, e.g., RoBERTa)
        model_inputs = {
            k: v.to(device)
            for k, v in enc.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"] and k in enc
        }

        with torch.no_grad():
            out = model(**model_inputs)
            start = out.start_logits.detach().cpu().numpy()
            end = out.end_logits.detach().cpu().numpy()

        # Collect per-feature predictions and null scores
        feature_predictions: List[Tuple[int, str, float]] = []  # (sample_idx, text, score)
        feature_null_scores: Dict[int, float] = {}              # sample_idx -> best null score

        overflow_to_sample = enc["overflow_to_sample_mapping"]
        for fi in range(len(enc["input_ids"])):
            si = int(overflow_to_sample[fi])  # which example in this micro-batch
            offsets = enc["offset_mapping"][fi].tolist()
            seq_ids = enc.sequence_ids(fi)

            # Valid context token positions
            valid_positions = [j for j, (sid, off) in enumerate(zip(seq_ids, offsets)) if sid == 1 and off is not None]
            if not valid_positions:
                continue

            s_logits = start[fi]
            e_logits = end[fi]

            # CLS/null usually at position 0
            null_score = float(s_logits[0] + e_logits[0])
            if (si not in feature_null_scores) or (null_score > feature_null_scores[si]):
                feature_null_scores[si] = null_score

            # Best span under constraints
            bs_idx, be_idx, best_score = best_span_from_logits(s_logits, e_logits, qa_max_answer_len)
            bs_idx = max(bs_idx, valid_positions[0])
            be_idx = min(be_idx, valid_positions[-1])
            if be_idx < bs_idx:
                be_idx = bs_idx

            (char_s, _) = offsets[bs_idx]
            (_, char_e) = offsets[be_idx]
            context_text = batch[si]["context"]
            pred_text = context_text[char_s:char_e].strip()

            feature_predictions.append((si, pred_text, float(best_score)))

        # Reduce features → per-example
        per_example_best_span: Dict[int, Tuple[str, float]] = {}
        for si, text, score in feature_predictions:
            if (si not in per_example_best_span) or (score > per_example_best_span[si][1]):
                per_example_best_span[si] = (text, score)

        # Decide span vs null for each example in the micro-batch
        batch_answers = ["unanswerable"] * len(batch)
        for local_idx in range(len(batch)):
            best_span_text, best_span_score = per_example_best_span.get(local_idx, ("", -1e9))
            null_score = feature_null_scores.get(local_idx, -1e9)

            # SQuAD v2 rule: if null - best_span > threshold → unanswerable
            if (null_score - best_span_score) > na_threshold or not best_span_text:
                batch_answers[local_idx] = "unanswerable"
            else:
                batch_answers[local_idx] = best_span_text if best_span_text.strip() else "unanswerable"

        answers.extend(batch_answers)

        done = min(i + batch_size, n)
        if (i // batch_size) % 10 == 0 or done == n:
            print(f"[Progress] {done}/{n}", file=sys.stderr, flush=True)

    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/brachmat/phd/models/bert-base-uncased")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_pred", default="predictions_squad_bert.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=5)

    parser.add_argument("--qa_max_length", type=int, default=384, help="Tokenizer max length.")
    parser.add_argument("--qa_doc_stride", type=int, default=128, help="Sliding window stride.")
    parser.add_argument("--qa_max_answer_len", type=int, default=30, help="Max tokens for an answer span.")
    parser.add_argument("--na_threshold", type=float, default=0.0,
                        help="Predict 'unanswerable' if (null_score - best_span_score) > threshold.")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    ds = load_dataset("/home/brachmat/phd/datasets/squad_v2", split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"[Info] Limiting to {len(ds)} examples for a quick test.")
    else:
        print(f"[Info] Using full dataset: {len(ds)} examples")

    examples = to_examples(ds)
    tokenizer, model, device = load_qa_model(args.model)

    answers = run_qa(
        tokenizer, model, device, examples,
        qa_max_length=args.qa_max_length,
        qa_doc_stride=args.qa_doc_stride,
        qa_max_answer_len=args.qa_max_answer_len,
        na_threshold=args.na_threshold,
        batch_size=args.batch_size
    )

    # Same JSONL schema as before
    with open(args.output_pred, "w", encoding="utf-8") as fout:
        for ex, pred in zip(examples, answers):
            golds = ex["golds"]
            gold_one = golds[0] if golds else ""
            fout.write(json.dumps({
                "id": ex["qid"],
                "context": ex["context"],
                "question": ex["question"],
                "predicted_answer": pred,
                "gold_answer": gold_one
            }, ensure_ascii=False) + "\n")

    print("✅ Done. Predictions written to", args.output_pred)

if __name__ == "__main__":
    main()
