import argparse
import json
import os
import random
from typing import List, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name_or_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def build_prompt(post: str) -> str:
    return (
        "Please write a short TL;DR summary for the following Reddit post. "
        "Keep only the main concern or question, and avoid unnecessary details.\n\n"
        f"Post:\n{post}\n\n"
        "TL;DR:"
    )


def generate_one(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 256,  # 减小默认值
    temperature: float = 0.1,    # 降低温度
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,  # 添加重复惩罚
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = temperature > 0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,  # 惩罚重复
            no_repeat_ngram_size=3,  # 禁止重复3-gram
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    
    # 如果生成太短或为空，返回提示
    if len(text) < 10:
        return "[Model generated very short output]"
    
    return text

def pick_post(example: Dict) -> str:
    # 尽量兼容几个可能的数据字段名
    for key in ["post", "content", "article", "document", "text", "prompt"]:
        if key in example and example[key]:
            return str(example[key]).strip()
    raise KeyError(f"Cannot find post field in example keys: {list(example.keys())}")


def load_eval_subset(dataset_name: str, split: str, num_samples: int, seed: int):
    ds = load_dataset(dataset_name, split=split)
    if num_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_samples))
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--drpo_model", type=str, default="output/tldr")
    parser.add_argument("--dataset_name", type=str, default="trl-lib/tldr")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_file", type=str, default="compare_tldr_outputs.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading base model: {args.base_model}")
    base_tokenizer, base_model = load_model_and_tokenizer(args.base_model, device)

    print(f"Loading DRPO model: {args.drpo_model}")
    drpo_tokenizer, drpo_model = load_model_and_tokenizer(args.drpo_model, device)

    print(f"Loading dataset: {args.dataset_name} [{args.split}]")
    ds = load_eval_subset(args.dataset_name, args.split, args.num_samples, args.seed)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    results: List[Dict] = []

    for i, ex in enumerate(ds):
        post = pick_post(ex)
        prompt = build_prompt(post)

        try:
            base_output = generate_one(
                base_model,
                base_tokenizer,
                prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            base_output = f"[ERROR] {repr(e)}"

        try:
            drpo_output = generate_one(
                drpo_model,
                drpo_tokenizer,
                prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            drpo_output = f"[ERROR] {repr(e)}"

        row = {
            "index": i,
            "prompt": prompt,
            "post": post,
            "base_model": args.base_model,
            "base_output": base_output,
            "drpo_model": args.drpo_model,
            "drpo_output": drpo_output,
        }
        results.append(row)

        print("=" * 80)
        print(f"[{i}]")
        print("BASE OUTPUT:")
        print(base_output)
        print("-" * 40)
        print("DRPO OUTPUT:")
        print(drpo_output)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved to: {args.output_file}")


if __name__ == "__main__":
    main()