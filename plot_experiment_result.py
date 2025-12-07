import json
import matplotlib.pyplot as plt
import os

json_dir = "./experiment_results/train"   # 你的 json 文件夹路径

plt.figure()

for file_name in os.listdir(json_dir):
    if file_name.endswith(".json"):
        file_path = os.path.join(json_dir, file_name)

        with open(file_path, "r") as f:
            data = json.load(f)

        steps = [item[1] for item in data]
        losses = [item[2] for item in data]

        label = file_name.replace(".json", "")
        plt.plot(steps, losses, label=label)

plt.xlabel("Training Step", fontsize=15)
plt.ylabel("Training Loss", fontsize=15)
plt.xlim(0, 2000) 
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.grid(True)
# ✅ 1. 定义输出文件路径变量
output_file_path = "plot_results/train_loss_curve.pdf"

# ✅ 2. 自动创建目录（如果不存在）
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# ✅ 3. 保存图片

plt.tight_layout()
plt.savefig(output_file_path, dpi=300)
plt.show()
