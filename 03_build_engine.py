# 03_build_engine.py —— ONNX 编译成 TensorRT engine（TRT 10.x 写法）
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

# TRT10：显式 batch 是默认，直接 create_network()
network = builder.create_network()

parser = trt.OnnxParser(network, logger)
with open("resnet50.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print("解析错误:", parser.get_error(i))
        raise SystemExit("ONNX 解析失败")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

# 动态 batch：告诉 TRT batch 范围
profile = builder.create_optimization_profile()
profile.set_shape("input", (1, 3, 224, 224), (1, 3, 224, 224), (8, 3, 224, 224))
config.add_optimization_profile(profile)

print("正在编译 TensorRT 引擎，请耐心等待（几十秒~几分钟）...")
serialized = builder.build_serialized_network(network, config)
with open("resnet50.engine", "wb") as f:
    f.write(serialized)
print("编译完成 → resnet50.engine")
