from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 配置你的本地模型路径
MODEL_PATH = "path/to/your/local/model"  # 改成你的模型路径

class LocalModel:
    def __init__(self, model_path):
        print(f"正在加载模型: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # 半精度节省显存
            device_map="auto",           # 自动分配设备
            trust_remote_code=True
        )
        
        # 如果没有 pad_token，设置成 eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("模型加载完成！")
    
    def ask(self, prompt, max_new_tokens=512, temperature=0.7):
        """
        向模型提问
        
        参数:
            prompt: 输入的问题/提示
            max_new_tokens: 最大生成 token 数
            temperature: 随机性 (0~1)
        """
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,          # 开启采样
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出（只保留新生成的部分）
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return response

# 主程序
if __name__ == "__main__":
    # 初始化模型
    model = LocalModel(MODEL_PATH)
    
    print("\n" + "=" * 50)
    print("本地模型测试程序")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 50)
    
    while True:
        prompt = input("\n[你]: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        
        if not prompt:
            continue
        
        print("\n[模型]: ", end="", flush=True)
        response = model.ask(prompt)
        print(response)