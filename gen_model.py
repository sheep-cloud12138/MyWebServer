import torch
import torch.nn as nn

# 1. 定义一个超级简单的神经网络：只有一个加法操作
class SimpleAdd(nn.Module):
    def forward(self, x):
        return x + 100.0

# 2. 实例化模型
model = SimpleAdd()
model.eval()

# 3. 定义虚拟输入 (BatchSize=1, Dimension=1)
dummy_input = torch.randn(1, 1)

# 4. 导出为 ONNX 格式
# 注意：input_names 和 output_names 必须和 C++ 代码里的对应！
# 我们在 C++ 里写死了 "input" 和 "output"
torch.onnx.export(model, 
                  dummy_input, 
                  "test_model.onnx", 
                  input_names=['input'], 
                  output_names=['output'],
                  opset_version=17)

# 5. 修复 IR version 兼容性 (onnxruntime 1.16.3 最高支持 IR version 9)
import onnx
onnx_model = onnx.load("test_model.onnx")
if onnx_model.ir_version > 9:
    print(f"Fixing IR version from {onnx_model.ir_version} to 8 for onnxruntime 1.16.3 compatibility")
    onnx_model.ir_version = 8
    onnx.save(onnx_model, "test_model.onnx")

print("Model saved to test_model.onnx! Expect output = input + 100")