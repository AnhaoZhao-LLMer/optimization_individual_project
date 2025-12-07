from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/code/models/R1-Distill-Qwen-1.5B/"          # 例: "meta-llama/Llama-3-8B"
ADAPTER_DIR = "/code/optimization_project/SFT-Train/sgd/checkpoint-1800"    

# 1. 加载 base model 和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/code/models/R1-Distill-Qwen-1.5B/")

# 2. 加载并合并 adapter
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

# 3. 把 adapter 合并进 base model 权重中
model = model.merge_and_unload()

# 4. 保存合并后的完整模型
output_dir = "./result_model/sgd"
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

print(f"✅ 合并完成，已保存到: {output_dir}")
