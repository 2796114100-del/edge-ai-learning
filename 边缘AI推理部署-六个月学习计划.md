# 边缘AI推理部署工程师 — 六个月详细学习计划

> **核心理念**：不系统学，直接上手干。每阶段只学当前项目需要的东西，够用就停，立刻动手写代码。目标不是"学完"，是"跑通"。

> 📚 **这份是最初的高层草图，仅供快速浏览。真正上课用 `边缘AI学习教程/` 文件夹**（经 web 调研重写、细化到每天每节，且已更新决策）。
> 每天学习：让 AI 打开 `边缘AI学习教程/00-教学总纲.md`（教学规则）+ 当天阶段文件（完整代码/检查点/报错对照/适配提示）。
>
> ⚠️ **下面草图里的技术细节已被教程包更新，以教程包为准**：检测模型改用 **YOLOv8**（非 v5）；Jetson 固定 **JetPack 6.2**；TensorRT 插件用 **IPluginV3**（非 v2）；阶段一用 torch `.data_ptr()` 免装 pycuda。

---

## 你的硬件情况

| 设备 | 型号 | 用途 |
|------|------|------|
| PC GPU | **Quadro RTX 3000（6GB）** | 第一阶段 TensorRT 推理够用，不能做训练 |
| 训练 | **租云服务器** | AutoDL / 矩池云，按小时租 A100 |
| Jetson | **Orin Nano 8GB（立刻下单）** | 第二~五阶段部署平台 |

> CUDA 架构：PC 是 SM 7.5 → 编译用 `-arch=sm_75`；Jetson Orin 是 SM 8.7 → 编译用 `-arch=sm_87`。

### 立刻下单 Jetson

不管第一阶段做什么，**Jetson 今天下单**，到货加刷机要一周，别让它成为阻塞项。

| 硬件 | 推荐型号 | 价格 |
|------|---------|------|
| 核心板 | Jetson Orin Nano 8GB（67 TOPS） | ~2000 元 |
| microSD 卡 | 128GB UHS-1（闪迪/三星） | ~80 元 |
| 电源 | 5V 4A DC 电源适配器 | ~50 元 |
| 摄像头 | IMX219 CSI 摄像头 | ~100 元 |
| 外壳 | 亚克力外壳+风扇 | ~80 元 |
| 串口线 | USB 转 TTL（CH340） | ~20 元 |
| 下位机 | Arduino Nano + 舵机 | ~30 元 |
| **总计** | | **~2360 元** |

---

## 第一阶段：跑通第一个推理程序（第 1-2 周）

### 目标
把 PyTorch 分类模型导出成 ONNX，用 TensorRT 跑通推理，对比 PyTorch → ONNX Runtime → TensorRT 三者的推理速度。

### B 站视频

| 内容 | 搜索关键词 | 说明 |
|------|-----------|------|
| PyTorch 导出 ONNX | `PyTorch ONNX export 教程` | 看一个就行，10 分钟 |
| TensorRT 入门 | `TensorRT 入门教程` | 推荐看 NVIDIA 官方中文的 |

### 动手步骤

**第 1 天：装环境**

```powershell
# 确认 CUDA 版本（你的驱动 573 支持 CUDA 12.8）
nvidia-smi

# 装 PyTorch（CUDA 12.4 版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 装 TensorRT（从 NVIDIA 官网下 wheel，注意版本要对齐 CUDA）
# https://developer.nvidia.com/tensorrt/download
pip install tensorrt-cu12

# 验证
python -c "import torch; print(torch.cuda.is_available())"
python -c "import tensorrt; print(tensorrt.__version__)"
```

**第 2 天：PyTorch → ONNX**

```python
import torch
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
model = model.cuda().eval()
dummy_input = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(model, dummy_input, "resnet50.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}})
```

验证 ONNX：

```python
import onnx
model_onnx = onnx.load("resnet50.onnx")
onnx.checker.check_model(model_onnx)  # 不报错就对
```

**第 3 天：PyTorch 直接推理 → ONNX Runtime 推理 → TensorRT 推理 → 速度对比**

```python
# ONNX Runtime（CPU 推理当基准）
import onnxruntime as ort
session = ort.InferenceSession("resnet50.onnx")
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {"input": dummy})
```

```python
# TensorRT Engine 构建（跑一次就行，Engine 存下来复用）
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open("resnet50.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
serialized = builder.build_serialized_network(network, config)
with open("resnet50.engine", "wb") as f:
    f.write(serialized)
```

```python
# TensorRT 推理（加载 Engine → 分配显存 → 推理）
runtime = trt.Runtime(logger)
with open("resnet50.engine", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
# ... allocate buffers, memcpy, enqueueV3
```

**第 4-5 天：速度对比表**

| 框架 | 推理时间 | 相对 PyTorch 提升 |
|------|---------|------------------|
| PyTorch（GPU） | XX ms | 基准 |
| ONNX Runtime（CPU） | XX ms | XX% |
| ONNX Runtime（GPU） | XX ms | XX% |
| TensorRT（FP32） | XX ms | XX% |

### 输出
- `resnet50_to_onnx.py` — 导出脚本
- `resnet50_infer_onnxruntime.py` — ONNX Runtime 推理
- `resnet50_infer_tensorrt.py` — TensorRT 推理
- `benchmark.md` — 速度对比表

### 原则
不碰量化，不碰 C++。只用 Python，只看 FP32。

### 会踩的坑
- `No module named 'tensorrt'` → Windows 上装 TensorRT 要先装 CUDA + cuDNN，然后从 NVIDIA 官网下 TensorRT wheel。嫌麻烦的话，这步直接跳过，留到 Jetson 上做。
- `ONNX export failed: Unsupported operator` → ResNet50 是标准模型不会遇到，以后遇到别的模型再查。

---

## 第二阶段：Jetson 部署 YOLOv5（第 3-6 周）

### 目标
YOLOv5 在 Jetson Orin Nano 上跑到实时（15 FPS+），掌握 FP16/INT8 量化。

### B 站视频

| 内容 | 搜索关键词 | 说明 |
|------|-----------|------|
| Jetson 刷机 | `Jetson Orin Nano 刷机 JetPack` | 看官方 NVIDIA 嵌入式 频道的教程 |
| CSI 摄像头点亮 | `Jetson CSI 摄像头 测试` | 5 分钟搞定 |
| YOLOv5 TensorRT 部署 | `YOLOv5 TensorRT Jetson` | 找播放量高的，看两三个对比 |
| TensorRT 量化 INT8 | `TensorRT INT8 量化 校准` | 重点看 ptq calibration |

### 动手步骤

**第 3 周：刷机 + 环境搭建**

1. 按 B 站教程给 Jetson 刷 JetPack（JetPack 6.x 自带 CUDA + TensorRT + cuDNN）
2. 验证环境：
```bash
nvcc --version          # CUDA
python3 -c "import tensorrt; print(tensorrt.__version__)"  # TensorRT
```
3. 点亮 CSI 摄像头：
```bash
# 测试摄像头是否识别
ls /dev/video*
# 拍一张测试
nvgstcapture-1.0 --capture-auto
```

**第 4 周：YOLOv5 导出 + TRT 推理**

```python
# 1. 导出 YOLOv5 为 ONNX
# 在 PC 上跑（Jetson 也能跑，但 PC 更快）
!git clone https://github.com/ultralytics/yolov5
!cd yolov5 && python export.py --weights yolov5s.pt --include onnx
```

```python
# 2. Jetson 上用 trtexec 转 TensorRT Engine
# FP32 版本
trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s_fp32.engine

# FP16 版本
trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s_fp16.engine --fp16

# INT8 版本（需要校准数据）
trtexec --onnx=yolov5s.onnx \
        --saveEngine=yolov5s_int8.engine \
        --int8 \
        --calib=calibration_cache.bin  # 先要生成校准缓存
```

```python
# 3. Python 推理脚本（你只需要实现这个函数）
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def inference(image, engine, context, inputs, outputs, bindings):
    """
    image: numpy array, shape (H, W, 3), BGR
    engine: TensorRT engine
    返回: [x1, y1, x2, y2, conf, cls] 的列表
    """
    # 前处理：resize + normalize + HWC→CHW
    img = cv2.resize(image, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = img.astype(np.float32) / 255.0
    img = np.ascontiguousarray(img)

    # 拷贝到 GPU
    cuda.memcpy_htod(inputs[0]['device'], img)

    # 执行推理
    context.execute_v2(bindings)

    # 拷贝结果回 CPU
    cuda.memcpy_dtoh(outputs[0]['host'], outputs[0]['device'])

    # 后处理：解析输出 → NMS → 画框
    boxes = postprocess(outputs[0]['host'])
    return boxes
```

**第 5 周：量化专项（重点）**

这是面试高频考察点，花 2-3 天专门做。

```python
# INT8 校准脚本核心逻辑
import tensorrt as trt

# 1. 准备校准数据集（从训练集/验证集抽 500-1000 张图）
class CalibrationDataLoader:
    def __init__(self, image_list):
        self.images = image_list
        self.index = 0

    def get_batch(self, names):
        if self.index >= len(self.images):
            return None
        batch = []
        for _ in range(len(names)):
            img = cv2.imread(self.images[self.index])
            img = preprocess(img)  # 跟推理时同样的前处理
            batch.append(img)
            self.index += 1
        return batch

# 2. 用 trtexec 或者 Python API 做 calibrated INT8
# 最简单的方式：
# trtexec --onnx=yolov5s.onnx --int8 --calib=calib.cache \
#         --saveEngine=yolov5s_int8.engine
```

量化对比表（你需要在 Jetson 上实测）：

| 精度 | 推理速度 | mAP@0.5 | 精度损失 | 显存占用 |
|------|---------|---------|---------|---------|
| FP32 | XX ms | XX% | 基准 | XX MB |
| FP16 | XX ms | XX% | XX% | XX MB |
| INT8 | XX ms | XX% | XX% | XX MB |

**面试话术准备（量化部分）**：
- "INT8 校准用的是 entropy 方法，校准数据集用了 500 张 COCO 验证集图片"
- "FP16 精度基本不丢，速度提升约 1.8 倍；INT8 精度掉了 2.3 个点，但速度提升了 3 倍"
- "PTQ 够用就不用 QAT，PTQ 掉精度太狠才考虑 QAT"

**第 6 周：性能基准 + 写报告**

```
deploy_report.md
├── 环境版本（JetPack/CUDA/TensorRT 版本号）
├── 踩坑记录（每天遇到的问题和解决方案）
└── 性能数据（FP32/FP16/INT8 速度+精度+显存对比表格）
```

### 输出
- `yolov5_inference.py` — 完整推理脚本（含前处理+后处理+画框）
- `quantization_benchmark.py` — 量化对比脚本
- `deploy_report.md` — 部署报告
- Demo 视频：摄像头实时检测画面

### 会踩的坑
- 电源不够 5V 4A → 直接买指定电源，别用手机充电头
- CSI 排线金手指朝向 → 插之前看 B 站拆机视频对照
- `trtexec: command not found` → `/usr/src/tensorrt/bin/` 下面找
- INT8 精度掉太多 → 校准数据集不够多/不够有代表性，加到 1000 张试试

---

## 第三阶段：C++ 推理 + 自定义 TensorRT 插件（第 7-11 周）

### 目标
- 能用 C++ 调用 TensorRT API 做推理
- 把 NMS 写成 CUDA 核函数 + TensorRT 插件，编译成 .so，Python 加载

> ⚠️ **这是整个计划最难的部分**。前两周你可能会很挫败，正常。

### B 站视频

| 内容 | 搜索关键词 | 说明 |
|------|-----------|------|
| C++ 快速入门 | `C++ 入门教程 黑马程序员` | 只看到"类和对象"那一章 |
| CMake 入门 | `CMake 教程` | 看 20 分钟那个就行 |
| CUDA 编程入门 | `CUDA 编程 入门 核函数` | 看 grid/block/thread 的概念 |
| TensorRT 插件开发 | `TensorRT Plugin 开发` | 中文资源少，**看英文也不行就看 NVIDIA 官方 Sample** |

### 第 7 周：C++ 快速补课（只学够用的）

> 不用看完整课程，用到什么学什么。

**必须会的东西（对着编辑器写，不能用 IDE 自动补全偷懒）**：

```cpp
// 1. 智能指针
#include <memory>
std::shared_ptr<int> a = std::make_shared<int>(42);
std::unique_ptr<int> b = std::make_unique<int>(10);

// 2. STL 容器
#include <vector>
#include <map>
#include <string>
std::vector<float> boxes = {1.0, 2.0, 3.0};
std::map<std::string, int> config = {{"width", 640}, {"height", 640}};

// 3. 类继承 + 虚函数
class BasePlugin {
public:
    virtual int enqueue() = 0;  // 子类必须实现
};
class MyPlugin : public BasePlugin {
public:
    int enqueue() override { return 0; }
};

// 4. extern "C" 导出（让 Python 能调 .so）
extern "C" {
    void* create_plugin() { return new MyPlugin(); }
}

// 5. CMakeLists.txt 基础
cmake_minimum_required(VERSION 3.10)
project(my_plugin)
find_package(CUDA REQUIRED)
add_library(my_plugin SHARED src/plugin.cpp src/kernel.cu)
target_link_libraries(my_plugin nvinfer)  # TensorRT 库
```

**需要装的东西**：
- VS Code 或 CLion（写 C++，推荐 CLion 因为 CMake 集成好）
- CMake（装好就行）
- CUDA Toolkit（跟 Jetson 上版本对齐）

### 第 8 周：C++ 调用 TensorRT 做推理

**先不写插件**，先用 C++ 调 TensorRT 的 C++ API 把 YOLOv5 跑通。这步是过渡，让你熟悉 C++ 侧的 TensorRT 生态：

```cpp
// 核心流程：跟 Python API 完全对应
#include "NvInfer.h"
#include "NvOnnxParser.h"

// 1. 创建 Builder
auto builder = nvinfer1::createInferBuilder(logger);

// 2. 创建网络 + ONNX Parser
auto network = builder->createNetworkV2(
    1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
auto parser = nvonnxparser::createParser(*network, logger);
parser->parseFromFile("yolov5s.onnx", ...);

// 3. 构建 Engine
auto config = builder->createBuilderConfig();
auto engine = builder->buildSerializedNetwork(*network, *config);

// 4. 推理
auto runtime = nvinfer1::createInferRuntime(logger);
auto deserialized = runtime->deserializeCudaEngine(...);
auto context = deserialized->createExecutionContext();
context->enqueueV2(buffers, stream, nullptr);
```

B 站搜索 **`TensorRT C++ 推理`**，找一个视频对着敲。

### 第 9-10 周：写 NMS Plugin

> **核心思路**：先抄 NVIDIA 官方的 SamplePlugin，把 NMS 逻辑替换进去。

NVIDIA 官方 Plugin 示例路径（Jetson 上）：
```bash
/usr/src/tensorrt/samples/python/samplePlugin/
```

**NMS CUDA 核函数（照着写，理解每行在干什么就行）**：

```cpp
// kernel.cu
__global__ void nms_kernel(
    float* boxes,       // [N, 4] x1,y1,x2,y2
    float* scores,      // [N] 置信度
    int num_boxes,
    float iou_threshold,
    bool* keep          // [N] 输出：哪些框保留
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_boxes) return;

    // 如果这个框已经被抑制了，跳过
    // 对每个分数比自己高的框，算 IOU，超过阈值就抑制
    for (int j = 0; j < num_boxes; j++) {
        if (scores[j] <= scores[idx]) continue;
        float iou = compute_iou(boxes + idx * 4, boxes + j * 4);
        if (iou > iou_threshold) {
            keep[idx] = false;
            return;
        }
    }
}

__device__ float compute_iou(float* a, float* b) {
    float x1 = max(a[0], b[0]);
    float y1 = max(a[1], b[1]);
    float x2 = min(a[2], b[2]);
    float y2 = min(a[3], b[3]);
    float inter = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (area_a + area_b - inter + 1e-6);
}

// 启动核函数
void launch_nms_kernel(float* boxes, float* scores, int num_boxes,
                       float iou_threshold, bool* keep, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_boxes + threads - 1) / threads;
    nms_kernel<<<blocks, threads, 0, stream>>>(boxes, scores, num_boxes,
                                                iou_threshold, keep);
}
```

**TensorRT Plugin 类（照着写，不要求完全理解）**：

```cpp
// yolov5_nms_plugin.cpp
class YOLOv5NMSPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    // ---- 必须实现的虚函数 ----
    int32_t getNbOutputs() const override { return 4; }  // boxes, scores, classes, num_detections

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) override {
        // 动态 shape：输出维度取决于检测到多少个目标
    }

    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* inputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) override {
        // 在这个函数里调用你的 CUDA 核函数
        launch_nms_kernel(...);
        return 0;
    }

    // ... 其他虚函数 （getOutputDataType、configurePlugin、clone 等）
};

// 注册插件（宏不能写错）
REGISTER_TENSORRT_PLUGIN(YOLOv5NMSPluginCreator);
```

**CMakeLists.txt（照着抄）**：

```cmake
cmake_minimum_required(VERSION 3.10)
project(yolov5_nms_plugin)

find_package(CUDA REQUIRED)

# PC 上用 sm_75（Quadro RTX 3000），Jetson 上用 sm_87（Orin）
set(CMAKE_CUDA_ARCHITECTURES 75)   # PC
# set(CMAKE_CUDA_ARCHITECTURES 87) # Jetson

add_library(yolov5_nms_plugin SHARED
    src/yolov5_nms_plugin.cpp
    src/kernel.cu
)

target_include_directories(yolov5_nms_plugin PRIVATE
    ${TensorRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(yolov5_nms_plugin
    nvinfer        # TensorRT 核心库
    cudart         # CUDA Runtime
)

# Python 加载这个 .so 就行了
```

**编译 + Python 调用验证**：

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
# 产出 libyolov5_nms_plugin.so
```

```python
# Python 加载插件
import ctypes
ctypes.cdll.LoadLibrary("./build/libyolov5_nms_plugin.so")
# 正常加载不报错 → 插件注册成功
# 然后正常用 TensorRT Python API 加载 ONNX，插件会自动生效
```

### 第 11 周：速度对比 + Debug

```python
# CPU 后处理 vs GPU 插件后处理 速度对比
import time

# CPU NMS
t0 = time.time()
boxes_cpu = torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold)
t_cpu = time.time() - t0

# GPU Plugin NMS（在 Engine 内部执行，不需要额外耗时）
# 总推理时间对比
t_gpu = total_inference_time_with_plugin

print(f"CPU NMS: {t_cpu*1000:.2f}ms, GPU Plugin: {t_gpu*1000:.2f}ms")
```

### 输出
- `yolov5_cpp_inference/` — C++ TensorRT 推理工程
- `yolov5_nms_plugin/` — NMS Plugin 完整源码 + CMakeLists.txt
- `benchmark_plugin.md` — CPU vs GPU Plugin 速度对比

### 会踩的坑
- `undefined reference to nvinfer1::...` → CMakeLists.txt 里没 link TensorRT 库
- CUDA 核函数结果全 0 → 检查 grid/block 维度，或者 cudaMemcpy 没拷贝对
- Plugin 加载失败 → REGISTER_TENSORRT_PLUGIN 宏写错了，或者 .so 没编出来
- 动态 shape 输出维度算错 → 抄官方 DynamicResize 插件的 getOutputDimensions

---

## 第四阶段：ROS2 节点 + 推理 + 串口控制（第 12-15 周）

### 目标
C++ ROS2 节点订阅相机话题 → 调 TensorRT 推理 → 串口发结果 → Arduino 控制舵机。

### B 站视频

| 内容 | 搜索关键词 | 说明 |
|------|-----------|------|
| ROS2 入门 | `ROS2 Humble 教程 古月居` 或 `赵虚左 ROS2` | 只看 Topic 通信那一章 |
| ROS2 C++ 节点 | `ROS2 C++ publisher subscriber` | 看怎么写 C++ 节点的 |
| Arduino 入门 | `Arduino 舵机控制 串口` | 10 分钟搞定 |

### 动手步骤

**第 12 周：ROS2 环境搭建 + 第一个节点**

```bash
# Jetson 上装 ROS2（JetPack 6.x 用 ROS2 Humble）
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

```bash
# 创建 ROS2 工作空间
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
```

```cpp
// src/inference_node/src/inference_node.cpp
// 你的第一个 ROS2 C++ 节点
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

class InferenceNode : public rclcpp::Node {
public:
    InferenceNode() : Node("inference_node") {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&InferenceNode::image_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "Inference Node Started");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        // 1. ROS Image 消息 → cv::Mat
        // 2. cv::Mat → TensorRT 输入
        // 3. 推理
        // 4. 结果 → 串口发出（第 13 周再加）
        RCLCPP_INFO(this->get_logger(), "Received frame: %dx%d",
                    msg->width, msg->height);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<InferenceNode>());
    rclcpp::shutdown();
    return 0;
}
```

**第 13 周：把 TensorRT 推理逻辑嵌进 ROS2 节点**

```cpp
// 把第二阶段写的 TensorRT 推理逻辑封装成一个类
class TensorRTInference {
public:
    TensorRTInference(const std::string& engine_path);
    std::vector<Detection> infer(const cv::Mat& image);
private:
    nvinfer1::IExecutionContext* context_;
    cudaStream_t stream_;
    // ... buffers
};
```

然后在 ROS2 节点的 callback 里调用它：

```cpp
void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv::Mat frame = ros_image_to_cv_mat(msg);  // 需要 cv_bridge
    auto detections = trt_inference_->infer(frame);
    // 画框，发结果
}
```

**第 14 周：串口通信**

```cpp
// Jetson 端串口发送
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

class SerialSender {
public:
    SerialSender(const std::string& port, int baudrate) {
        fd_ = open(port.c_str(), O_RDWR | O_NOCTTY);
        // 配置串口：波特率、数据位、停止位
        struct termios tty;
        tcgetattr(fd_, &tty);
        cfsetospeed(&tty, B115200);
        tcsetattr(fd_, TCSANOW, &tty);
    }

    void send(int x, int y) {
        // 简单协议：<x=120,y=200>\n
        char buf[64];
        int len = snprintf(buf, sizeof(buf), "<x=%d,y=%d>\n", x, y);
        write(fd_, buf, len);
    }

private:
    int fd_;
};
```

**第 15 周：Arduino 端（半小时搞定）**

```cpp
// Arduino 代码，不到 20 行
#include <Servo.h>

Servo pan_servo;   // 水平
Servo tilt_servo;  // 垂直

void setup() {
    Serial.begin(115200);
    pan_servo.attach(9);
    tilt_servo.attach(10);
}

void loop() {
    if (Serial.available()) {
        String data = Serial.readStringUntil('\n');
        int x = data.substring(data.indexOf("x=")+2, data.indexOf(",")).toInt();
        int y = data.substring(data.indexOf("y=")+2, data.indexOf(">")).toInt();
        pan_servo.write(map(x, 0, 640, 0, 180));
        tilt_servo.write(map(y, 0, 480, 0, 180));
    }
}
```

### 输出
- `inference_ros2_ws/` — ROS2 工作空间（`colcon build` 通过）
- `arduino_servo/` — Arduino 舵机控制代码
- Demo 视频：相机检测到目标 → 舵机跟踪

### 会踩的坑
- `colcon build` 失败 → 检查 package.xml 依赖声明
- `cv_bridge` 编译不过 → 用 `image_transport` 或者直接跳过，手动传 cv::Mat
- 串口权限问题 → `sudo usermod -a -G dialout $USER` 然后重启
- 数据粘包 → 加 `\n` 做分隔符

---

## 第五阶段：端侧大模型（第 16-18 周）

### 目标
在 Jetson 上编译 llama.cpp，跑通一个小 LLM（Qwen2.5-0.5B / TinyLlama），理解端侧大模型部署的基本链路。

### B 站视频

| 内容 | 搜索关键词 | 说明 |
|------|-----------|------|
| llama.cpp 编译 | `llama.cpp Jetson 部署` | 找 Jetson 上的编译教程 |
| GGUF 量化 | `GGUF 量化 格式 端侧` | 了解 Q4_0 / Q8_0 是什么意思 |

### 动手步骤

**第 16 周：编译 llama.cpp**

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CUDA=1 -j$(nproc)
# 验证
./llama-cli --version
```

**第 17 周：下载模型 + 转换 + 量化**

```bash
# 下载 Qwen2.5-0.5B（HuggingFace）
# 如果被墙 → 用 modelscope 或者开梯子
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

# 转 GGUF
python convert_hf_to_gguf.py Qwen2.5-0.5B-Instruct --outfile qwen2.5-0.5b-f16.gguf

# 量化
./llama-quantize qwen2.5-0.5b-f16.gguf qwen2.5-0.5b-q4_0.gguf Q4_0
./llama-quantize qwen2.5-0.5b-f16.gguf qwen2.5-0.5b-q8_0.gguf Q8_0
```

**第 18 周：跑起来 + 性能记录**

```bash
# 推理测试
./llama-cli -m qwen2.5-0.5b-q4_0.gguf -p "你好，介绍一下你自己" -n 128

# 记录显存占用（开另一个终端）
watch -n 1 nvidia-smi
```

性能记录表：

| 量化方式 | 模型大小 | 显存占用 | 推理速度 (tokens/s) | 回答质量 |
|---------|---------|---------|--------------------|---------|
| FP16 | XX MB | XX GB | XX t/s | 基准 |
| Q8_0 | XX MB | XX GB | XX t/s | 几乎无损 |
| Q4_0 | XX MB | XX GB | XX t/s | 略下降 |

### 输出
- `edge_llm_report.md` — 编译+量化+性能数据
- 一个能对话的端侧 Demo
- 可以录个短视频：Jetson 上离线运行大模型对话

### 会踩的坑
- `make LLAMA_CUDA=1` 找不到 CUDA → 检查 `nvcc --version`，或者不加 `LLAMA_CUDA=1` 直接 CPU 推理也行
- 下载模型被墙 → 开梯子 `export HTTPS_PROXY=http://127.0.0.1:7890`
- 显存不够 → 选更小的模型（Qwen2.5-0.5B 只有 1GB 左右）
- 回答质量差 → Q4_0 量化损失大是正常的，用 Q8_0 会好很多

---

## 第六阶段：求职准备（第 19-20 周）

### B 站视频

| 内容 | 搜索关键词 | 说明 |
|------|-----------|------|
| 简历怎么写 | `程序员 简历 项目经验 怎么写` | 看 2-3 个 |
| 面试技巧 | `AI 推理部署 面试题` | 了解高频问题 |

### 第 19 周：项目整理

**GitHub 仓库结构**：

```
github.com/yourname/
├── trt-resnet50/           # 第一阶段：ONNX + TensorRT 推理
│   └── README.md           # 项目介绍、怎么跑、性能数据
├── yolov5-jetson/          # 第二阶段：YOLOv5 部署 + 量化
│   └── README.md           # 含量化对比表
├── tensorrt-nms-plugin/    # 第三阶段：NMS Plugin
│   └── README.md           # CPU vs GPU 后处理速度对比
├── ros2-inference/         # 第四阶段：ROS2 节点
│   └── README.md           # 含 Demo 视频链接
└── edge-llm/               # 第五阶段：端侧大模型
    └── README.md           # 含量化性能表
```

**每个 README 必须包含**：
1. 做了什么（一句话）
2. 怎么做的（技术栈）
3. 遇到什么坑（真实的，面试官爱听）
4. 怎么解决的
5. 最终性能数据（数字说话）

**简历上的项目描述模板**：

```
端到端实时目标检测跟踪系统 | Jetson Orin Nano + TensorRT + ROS2
- 使用 PyTorch→ONNX→TensorRT 链路部署 YOLOv5，FP16 推理 Xms/帧
- 完成 FP16/INT8 三精度量化对比，INT8 速度提升 X 倍，精度损失仅 X%
- 自研 NMS TensorRT Plugin（C++ + CUDA），GPU 后处理比 CPU 快 X 倍
- 编写 ROS2 C++ 节点订阅相机消息调用推理，串口协议遥控 Arduino 舵机追踪
- 在 Jetson 上编译 llama.cpp 部署 Qwen2.5-0.5B，Q4_0 量化仅占用 X GB 显存
```

### 第 20 周：投递 + 刷面试题

**投递渠道**：
- BOSS 直聘 搜：`TensorRT`、`模型部署`、`边缘计算`、`AI 推理`、`Jetson`、`ONNX`
- 目标公司：黑芝麻智能、东风研发总院、武汉里得电力、华威科智能、中焙智能

**面试高频问题准备（每个能说 3 分钟）**：

| 问题 | 你的回答来源 |
|------|------------|
| PyTorch 模型怎么部署到端侧？ | 第二阶段 |
| TensorRT 构建 Engine 的流程？ | 第一阶段 + 第二阶段 |
| INT8 量化怎么做？精度掉了怎么办？ | 第二阶段量化专项 |
| 写过 CUDA 吗？做什么的？ | 第三阶段 NMS Plugin |
| ONNX 算子不支持怎么办？ | 查文档 + 写 Plugin |
| 部署中遇到的最难的问题是什么？ | 第三阶段动态 shape + CUDA 调试 |

### 输出
- 5 个带 README 的 GitHub 项目
- 一份量化数据支撑的简历
- BOSS 直聘已投递记录

---

## 附录 A：技术栈速查表

| 技术 | 学到什么程度 | 遇到问题去哪查 | 参考视频/B站 |
|------|------------|--------------|-------------|
| Python | 调 API，写脚本 | 官方文档 | — |
| C++ | 能写 Plugin，能编译工程 | [cppreference.com](https://cppreference.com) | `黑马程序员 C++` |
| CUDA | 能看懂/修改核函数 | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | `CUDA 编程入门` |
| CMake | 能编译别人的工程，会写基础 CMakeLists | [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/) | `CMake 教程` |
| PyTorch | 会导出 ONNX | `torch.onnx.export` 文档 | `PyTorch ONNX export` |
| TensorRT | C++ 和 Python API 都能用 | [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/) | `TensorRT 入门教程` |
| ROS2 | Publisher + Subscriber | [ROS2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html) | `古月居 ROS2 入门` |
| ONNX | 会导出、会检查、会看 netron | [ONNX docs](https://onnx.ai/onnx/) | `ONNX 入门教程` |

## 附录 B：C++ 只学这些（对照表）

```cpp
// ✅ 要学的
#include <memory>         // shared_ptr, unique_ptr
#include <vector>         // std::vector
#include <map>            // std::map
#include <string>         // std::string
#include <iostream>       // std::cout
#include <fstream>        // 读文件
virtual int func() = 0;   // 纯虚函数
int func() override;      // 重写虚函数
extern "C" { ... };       // C 导出
find_package(...)         // CMake 找包
target_link_libraries(...)  // CMake 链接

// ❌ 不需要学的
template<typename T>      // 模板编程
std::move / std::forward  // 移动语义
<algorithm> 全库          // 算法库（std::sort 会用就行）
<thread> / <mutex>        // 多线程（遇到了再学）
```

## 附录 C：命令速查

```bash
# Jetson 常用命令
sudo jtop                       # 监控 CPU/GPU/内存（先装 jetson-stats）
nvidia-smi                      # GPU 状态
ls /dev/video*                  # 查看摄像头设备
v4l2-ctl --list-devices         # 视频设备列表
trtexec --onnx=model.onnx --fp16 --saveEngine=model.engine  # ONNX → TRT Engine
watch -n 1 nvidia-smi           # 实时监控 GPU

# 串口
sudo usermod -a -G dialout $USER  # 给串口权限
ls /dev/ttyUSB*                   # 查看串口设备

# 编译
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 附录 D：每周时间分配建议

假设每周能投入 15-20 小时（周一到周五晚上 2 小时 + 周末 5 小时）：

| 时间 | 做什么 |
|------|--------|
| 周一到周五 | 敲代码、Debug、跑实验 |
| 周六 | 学新东西（看 B 站、读文档） |
| 周日 | 写笔记 + 整理本周产出 + 更新 GitHub |

---

> **最后提醒**：遇到问题先 Google，再 Stack Overflow，再 NVIDIA 官方论坛。B 站教程三天出不来就别死磕了，换个关键词重新搜。六个月之后回头看，你会发现起步时最大的困难其实都不算事。
