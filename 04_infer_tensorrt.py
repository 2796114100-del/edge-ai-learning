# 04_infer_tensorrt.py —— 加载 engine 推理（TRT 张量名 API + torch 显存）
import torch, tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
with open("resnet50.engine", "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# 1. 找输入/输出张量名（TRT10 用名字，不用编号）
in_name = out_name = None
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        in_name = name
    else:
        out_name = name
print("输入张量:", in_name, "| 输出张量:", out_name)

# 2. 固定本次输入形状
context.set_input_shape(in_name, (1, 3, 224, 224))
out_shape = tuple(context.get_tensor_shape(out_name))

# 3. 直接用 torch 的 CUDA 张量当显存缓冲（不需要 pycuda！）
d_in = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float32)  # 随机输入
d_out = torch.empty(out_shape, device="cuda", dtype=torch.float32)      # 接收容器

# 4. 把张量的显存地址告诉 TensorRT
context.set_tensor_address(in_name, d_in.data_ptr())
context.set_tensor_address(out_name, d_out.data_ptr())

# 5. 在一个 CUDA stream 上执行推理
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    context.execute_async_v3(stream_handle=stream.cuda_stream)
stream.synchronize()

print("输出 shape:", tuple(d_out.shape))
print("预测类别编号:", int(d_out.argmax()))
