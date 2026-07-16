# 05_benchmark.py —— PyTorch vs TensorRT 速度对比
import time, torch, torchvision, tensorrt as trt
N = 100

# ---- A. PyTorch ----
model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.DEFAULT).eval().cuda()
x = torch.randn(1, 3, 224, 224, device="cuda")
for _ in range(10):                          # 预热 10 次
    with torch.no_grad(): model(x)
torch.cuda.synchronize()
t = time.time()
for _ in range(N):                           # 正式测 100 次
    with torch.no_grad(): model(x)
torch.cuda.synchronize()
pt_ms = (time.time() - t) / N * 1000

# ---- B. TensorRT ----
logger = trt.Logger(trt.Logger.WARNING)
with open("resnet50.engine", "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()
inn, outn = engine.get_tensor_name(0), engine.get_tensor_name(1)
ctx.set_input_shape(inn, (1, 3, 224, 224))
d_in = torch.randn(1, 3, 224, 224, device="cuda")
d_out = torch.empty(tuple(ctx.get_tensor_shape(outn)), device="cuda")
ctx.set_tensor_address(inn, d_in.data_ptr())
ctx.set_tensor_address(outn, d_out.data_ptr())
s = torch.cuda.Stream()
def run():
    with torch.cuda.stream(s):
        ctx.execute_async_v3(stream_handle=s.cuda_stream)
    s.synchronize()
for _ in range(10): run()                    # 预热 10 次
t = time.time()
for _ in range(N): run()                     # 正式测 100 次
trt_ms = (time.time() - t) / N * 1000

print(f"PyTorch (FP32):  {pt_ms:.2f} ms/张")
print(f"TensorRT (FP32): {trt_ms:.2f} ms/张")
print(f"加速比: {pt_ms / trt_ms:.2f} 倍")
