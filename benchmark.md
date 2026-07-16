# ResNet50 推理速度对比（Quadro RTX 3000, FP32）

| 框架 | 推理耗时 | 加速比 |
|------|---------|--------|
| PyTorch 原生 | 10.82 ms | 1.0x |
| TensorRT | 3.37 ms | 3.21x |

环境：Windows 11 + CUDA 13.2 + TensorRT 11.1。结论：不损失精度下 TensorRT 加速约 3.2 倍。
