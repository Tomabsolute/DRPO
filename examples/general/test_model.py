import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ====== Edit these settings directly ======
MODEL_PATH = "output/general_smoke"
USE_CHAT_TEMPLATE = True
MAX_NEW_TOKENS = 256
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9
USE_BF16 = True
# ==========================================


def format_prompt(tokenizer, user_text: str) -> str:
    if not USE_CHAT_TEMPLATE:
        return user_text
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and USE_BF16:
        dtype = torch.bfloat16
    elif device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    print("Interactive test started. Type 'exit' to quit.")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"exit", "quit", "q"}:
            break
        if not user_text:
            continue

        prompt_text = format_prompt(tokenizer, user_text)
        encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE if DO_SAMPLE else 1.0,
                top_p=TOP_P if DO_SAMPLE else 1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0, input_len:]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print(f"Model: {answer}")


if __name__ == "__main__":
    main()
