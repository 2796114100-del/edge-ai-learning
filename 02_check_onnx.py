# 02_check_onnx.py —— 检查 ONNX 合法 + 输出和 PyTorch 一致
import onnx, numpy as np, onnxruntime as ort, torch, torchvision

onnx.checker.check_model(onnx.load("resnet50.onnx"))
print("ONNX 结构检查通过 ✅")

model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.DEFAULT).eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    ref = model(x).numpy()

sess = ort.InferenceSession("resnet50.onnx", providers=["CPUExecutionProvider"])
out = sess.run(None, {"input": x.numpy()})[0]

print("两者最大误差:", float(np.abs(ref - out).max()))
print("结果一致 ✅" if np.allclose(ref, out, atol=1e-3) else "不一致 ❌")
