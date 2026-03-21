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


def build_hh_prompt(prompt_text: str, tokenizer) -> str:
    """
    HH 数据通常已经有较完整的 prompt。
    优先直接使用；如果 tokenizer 有 chat template，也可按需包一层。
    """
    prompt_text = str(prompt_text).strip()

    # 如果已经是 Human/Assistant 风格，直接返回
    if "Human:" in prompt_text or "Assistant:" in prompt_text:
        return prompt_text

    # 如果模型有 chat template，可以包装成 user 消息
    if getattr(tokenizer, "chat_template", None):
        try:
            messages = [{"role": "user", "content": prompt_text}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return prompt_text


def generate_one(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = temperature > 0

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if len(text) == 0:
        return "[Model generated empty output]"

    return text


def pick_hh_prompt(example: Dict) -> str:
    for key in ["prompt", "chosen", "rejected", "text"]:
        if key in example and example[key]:
            if key == "chosen" or key == "rejected":
                continue
            return str(example[key]).strip()
    raise KeyError(f"Cannot find prompt field in example keys: {list(example.keys())}")


def load_eval_subset(dataset_name: str, split: str, num_samples: int, seed: int):
    ds = load_dataset(dataset_name, split=split)
    if num_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_samples))
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Kyleyee/Qwen2.5-1.5B-sft-hh")
    parser.add_argument("--drpo_model", type=str, default="output/hh")
    parser.add_argument("--dataset_name", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_file", type=str, default="results/compare_hh_outputs.jsonl")
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
        raw_prompt = pick_hh_prompt(ex)
        base_prompt = build_hh_prompt(raw_prompt, base_tokenizer)
        drpo_prompt = build_hh_prompt(raw_prompt, drpo_tokenizer)

        chosen = ex.get("chosen", None)
        rejected = ex.get("rejected", None)

        try:
            base_output = generate_one(
                base_model,
                base_tokenizer,
                base_prompt,
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
                drpo_prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            drpo_output = f"[ERROR] {repr(e)}"

        row = {
            "index": i,
            "raw_prompt": raw_prompt,
            "base_prompt": base_prompt,
            "drpo_prompt": drpo_prompt,
            "chosen_reference": chosen,
            "rejected_reference": rejected,
            "base_model": args.base_model,
            "base_output": base_output,
            "drpo_model": args.drpo_model,
            "drpo_output": drpo_output,
        }
        results.append(row)

        print("=" * 80)
        print(f"[{i}]")
        print("PROMPT:")
        print(raw_prompt)
        print("-" * 40)
        if chosen is not None:
            print("CHOSEN REFERENCE:")
            print(chosen)
            print("-" * 40)
        if rejected is not None:
            print("REJECTED REFERENCE:")
            print(rejected)
            print("-" * 40)
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