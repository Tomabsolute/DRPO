import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ckpt = "./output/tldr"   # 或者 "./output/tldr/checkpoint-1000" 之类

tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
)
while True:
    prompt = input("请输入提示词：")
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(out[0], skip_special_tokens=True))

