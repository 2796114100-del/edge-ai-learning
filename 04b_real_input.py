# 04b_real_input.py —— 用真实图片验证推理 + 打印 top-5 类别
import torch, tensorrt as trt

# 加载 engine（和之前一样）
logger = trt.Logger(trt.Logger.WARNING)
with open("resnet50.engine", "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

in_name = out_name = None
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        in_name = name
    else:
        out_name = name

context.set_input_shape(in_name, (1, 3, 224, 224))
out_shape = tuple(context.get_tensor_shape(out_name))

# ---- 改这里：用 PyTorch 的预处理管线处理一张真正的图 ----
from torchvision import transforms
from PIL import Image
import json, urllib.request

# 用一张公开的测试图（金毛犬），走梯子下载
import os
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7892"
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
print("下载测试图片...")
proxy_handler = urllib.request.ProxyHandler({"https": "http://127.0.0.1:7892"})
opener = urllib.request.build_opener(proxy_handler)
urllib.request.install_opener(opener)
urllib.request.urlretrieve(url, "dog.jpg")
img = Image.open("dog.jpg").convert("RGB")

# ResNet50 的标准预处理：缩放 256 → 中心裁 224 → 转张量 → 归一化
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),                               # [0,255] → [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],                      # ImageNet 统计值
        std=[0.229, 0.224, 0.225]
    ),
])
d_in = preprocess(img).unsqueeze(0).cuda()               # 加 batch 维
# ----

d_out = torch.empty(out_shape, device="cuda", dtype=torch.float32)
context.set_tensor_address(in_name, d_in.data_ptr())
context.set_tensor_address(out_name, d_out.data_ptr())

stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    context.execute_async_v3(stream_handle=stream.cuda_stream)
stream.synchronize()

# ---- 打印 top-5 ----
scores = d_out.softmax(dim=1).cpu().squeeze()            # softmax 转概率
top5 = scores.topk(5)
print("\nTop-5 预测:")
for i, (idx, prob) in enumerate(zip(top5.indices, top5.values)):
    print(f"  {i+1}. 类别 {int(idx):4d}  — 置信度 {float(prob):.3%}")
