# 01_export_onnx.py —— ResNet50 导出为 ONNX
import torch, torchvision

print("加载 ResNet50（首次下载约 100MB）...")
model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.DEFAULT).eval().cuda()

dummy = torch.randn(1, 3, 224, 224).cuda()   # 1张图, 3通道, 224x224
torch.onnx.export(
    model, dummy, "resnet50.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
print("导出完成 → resnet50.onnx")
