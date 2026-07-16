# 03 · 阶段三：C++/CUDA + 自定义 NMS 插件（第 7-11 周）

> **给 AI 老师**：先读 `00-教学总纲.md`。**全程最难的一章，学生 C++/CUDA 接近零基础，做好心理建设：挫败感强是正常的。** 教学策略（经调研优化）：
> **① 把两个难点分开教**——先把 CUDA 核函数练到熟（这是真本事、面试要考），再单独搞 TensorRT 插件的"管道"，最后才合体。
> **② 每步都有"已知正确的参照物"可对比**——先跑通 TensorRT **内置的 EfficientNMS_TRT** 当基线，再自写、和它对答案，避免"到底是我核函数错还是插件接线错"的抓瞎。
> **③ 插件用 IPluginV3**（不是老的 IPluginV2！）——外壳样板基于成熟仓库改（用户已选此路），你负责把从零写的 CUDA 核塞进去。

**基准环境**：CUDA 12.x + **TensorRT 10.x**。PC(WSL/Ubuntu, sm_75) 上开发调试，再切 Jetson(sm_87)。

> 🚨 **给 AI 老师的头号提醒（版本，别教错）**：**`IPluginV2` / `IPluginV2DynamicExt` 在 TensorRT 10 已弃用，TensorRT 11 会移除。当前要用 `IPluginV3`（配 `IPluginCreatorV3One`）。** 网上几乎所有插件教程（含 wang-xinyu/tensorrtx）还是 V2——**看它们学"插件由哪些部件组成"的心智模型（这个不变），但新写的插件用 V3**。官方权威：
> - 自定义层（V3 推荐）：https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html
> - V3 的 C++ 类（IPluginV3OneCore/OneBuild/OneRuntime）：https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-cpp.html
> 不确定某个 V3 方法签名时，**明确告诉学生查上面官方页，别编。**

---

## 本阶段目标

```
补 C++(够用) → 从零写 CUDA 核函数(vectorAdd→IoU→NMS) → C++ 调 TensorRT 推理
→ 跑通内置 EfficientNMS(基线) → 写"空壳"IPluginV3 插件(打通管道) → 把自己的 NMS 核塞进插件 → 和基线对答案
```

**学完能做到**：能写/改 CUDA 核函数（面试硬通货）；能写 C++ TensorRT 推理；有一个自己做的 IPluginV3 插件 `.so`；能讲清插件机制。

**时间**：约 5 周。**这章慢没关系，质量比速度重要，多鼓励。**

---

## 参考资源（引用务必带【适配提示】）

| 资源 | 地址 | 用途 |
|------|------|------|
| NVIDIA "An Even Easier Introduction to CUDA" | https://developer.nvidia.com/blog/even-easier-introduction-cuda/ | **第一个核函数**的权威入门 |
| sangyc10 / CUDA-code | https://github.com/sangyc10/CUDA-code | 中文 B 站"CUDA编程基础入门系列"配套码 |
| codingonion / cuda-beginner-course | https://github.com/codingonion/cuda-beginner-course-cpp-version | CUDA 12.x 并行编程入门(C++) |
| leimao / TensorRT-Custom-Plugin-Example | https://github.com/leimao/TensorRT-Custom-Plugin-Example | **最佳"hello world"插件脚手架**（含 CMake） |
| wang-xinyu / tensorrtx | https://github.com/wang-xinyu/tensorrtx | 看真实插件由哪些部件组成（注意是 V2，学结构） |
| NVIDIA/TensorRT efficientNMSPlugin | https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin | 内置 NMS 源码（进阶研读） |

> 🔧 **【适配提示 · 通用】** ① CUDA 入门视频多在 x86 上，语法/核函数与平台无关，**能直接学**，只是编译时你在 Jetson 用 `sm_87`、PC(WSL) 用 `sm_75`。② 插件教程几乎都是 **IPluginV2**——**结构照学，接口用 V3**（见上方红字）。③ tensorrtx 用网络定义 API 手搭网络，和你"ONNX→引擎"路线不同，只看它的 `yololayer` 插件部分。

---

# 第 1 周 · C++ 速成 + 第一个 CUDA 核函数

> **给 AI 老师**：别让学生看完整 C++ 课（会陷进去）。只教下面够用的。强烈建议在 **WSL2(Ubuntu 22.04)** 里做，环境和 Jetson 一致。B站搜 `黑马程序员 C++`（只看到"类和对象/继承多态"）。

## Day 1 · 环境 + C++ 六件套

装：WSL2 Ubuntu + `sudo apt install build-essential cmake` + CUDA Toolkit。验证 `g++ --version`、`nvcc --version`、`cmake --version`。

`cpp_basics.cpp`（只学这 6 样，逐个跑）：

```cpp
#include <iostream>
#include <vector>
#include <memory>
using namespace std;

class Animal {                          // ④ 继承+虚函数（插件就是继承基类）
public:
    virtual void speak() = 0;           // 纯虚函数：子类必须实现
    virtual ~Animal() {}
};
class Dog : public Animal {
public:
    void speak() override { cout << "汪" << endl; }
};

int main() {
    int x = 42; cout << "x=" << x << endl;              // ① 变量/输出
    vector<float> v = {1,2,3}; v.push_back(4);          // ② STL 容器
    for (float e : v) cout << e << " "; cout << endl;
    shared_ptr<Dog> p = make_shared<Dog>();             // ③ 智能指针
    Animal* a = new Dog(); a->speak(); delete a;        // ④ 多态（TRT 调插件的原理）
    return 0;
}
```

```bash
g++ cpp_basics.cpp -o cpp_basics -std=c++14 && ./cpp_basics
```

✅ **【检查点 D1】** 输出 `x=42` / `1 2 3 4` / `汪`。报错 → C1。

> **【原理速讲】** 插件 = 继承 TensorRT 的基类 + 实现约定的虚函数。TensorRT 拿着基类指针调你的 `enqueue()`，不关心你怎么实现——就像 `Animal* a` 调 `a->speak()` 实际跑的是 `Dog` 的。第⑤样 `extern "C"` 和 CMake 用到再讲。

## Day 2 · 第一个 CUDA 核函数（vectorAdd）

> 跟 NVIDIA "Even Easier Introduction to CUDA" 做。

```cuda
// vector_add.cu —— 第一个核函数：两个数组相加
#include <cstdio>
__global__ void add(int n, float* a, float* b, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // "我是几号线程"
    if (idx < n) out[idx] = a[idx] + b[idx];           // 就处理几号数据
}
int main() {
    int n = 1000; size_t sz = n * sizeof(float);
    float *a, *b, *out;
    cudaMallocManaged(&a, sz); cudaMallocManaged(&b, sz); cudaMallocManaged(&out, sz);
    for (int i=0;i<n;i++){ a[i]=1.0f; b[i]=2.0f; }
    int threads=256, blocks=(n+threads-1)/threads;     // 向上取整覆盖所有元素
    add<<<blocks, threads>>>(n, a, b, out);
    cudaDeviceSynchronize();                           // 等 GPU 干完
    printf("out[0]=%.1f out[999]=%.1f\n", out[0], out[999]);   // 期望 3.0 3.0
    cudaFree(a); cudaFree(b); cudaFree(out);
    return 0;
}
```

```bash
nvcc vector_add.cu -o vector_add && ./vector_add
```

✅ **【检查点 D2】** 输出 `out[0]=3.0 out[999]=3.0`。报错 → C5。

> **【原理速讲】grid/block/thread（大白话）**：GPU 有几千小核心，适合"同一件事做很多遍"。**一个线程处理一个数据**；线程打包成 block（如 256 个），block 组成 grid。每个线程用 `blockIdx.x*blockDim.x+threadIdx.x` 算出"我是几号"。`cudaMallocManaged` = 统一内存，CPU/GPU 都能访问，省掉手动拷贝（入门友好）。

## Day 3-5 · 巩固 + `if(idx<n)` 的意义 + 缓冲

再写 1-2 个逐元素核（如数组乘标量、ReLU）。讲清 `if(idx<n)` 为什么必须有（线程总数常多于数据，多出的不能越界）。记笔记。

📝 **【小测 W1】** `add<<<blocks, threads>>>` 里两个数分别是什么？（答：block 数、每 block 线程数。）

---

# 第 2 周 · CUDA 核函数练到熟（IoU → NMS，暂不碰 TensorRT）

> **给 AI 老师**：这周是本阶段**最有含金量**的部分——学生亲手写出能用的 NMS 核并验证正确。慢慢来。

## Day 6-7 · reduction / argmax 核（NMS 要用到"找最大"）

写一个求数组最大值/argmax 的核（引入 `atomicMax` 或简单版），理解"多线程写同一个结果"的问题。（面试常考 reduce/softmax 核，这里打基础。）

## Day 8-9 · IoU + NMS 核函数（从零，逐行讲）

`nms_kernel.cu`：

```cuda
// nms_kernel.cu —— GPU 并行 NMS
#include <cuda_runtime.h>

__device__ float iou(const float* a, const float* b) {   // a,b = [x1,y1,x2,y2]
    float xx1=fmaxf(a[0],b[0]), yy1=fmaxf(a[1],b[1]);
    float xx2=fminf(a[2],b[2]), yy2=fminf(a[3],b[3]);
    float inter = fmaxf(0.f,xx2-xx1)*fmaxf(0.f,yy2-yy1);
    float areaA=(a[2]-a[0])*(a[3]-a[1]), areaB=(b[2]-b[0])*(b[3]-b[1]);
    return inter/(areaA+areaB-inter+1e-6f);
}
__global__ void nms_kernel(const float* boxes, const float* scores,
                           int n, float thr, bool* keep) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    keep[i] = true;
    for (int j=0;j<n;j++)                        // 存在"分更高且重叠超阈值"的框→我被抑制
        if (scores[j] > scores[i] && iou(&boxes[i*4], &boxes[j*4]) > thr) {
            keep[i] = false; return;
        }
}
extern "C" void launch_nms(const float* b, const float* s, int n, float thr, bool* keep) {
    int t=256, g=(n+t-1)/t;
    nms_kernel<<<g,t>>>(b,s,n,thr,keep);
    cudaDeviceSynchronize();
}
```

测试 `test_nms.cu`：

```cuda
#include <cuda_runtime.h>
#include <cstdio>
extern "C" void launch_nms(const float*,const float*,int,float,bool*);
int main(){
    float hb[]={10,10,50,50, 12,12,52,52, 100,100,140,140};   // 框0,1重叠;2独立
    float hs[]={0.9f,0.6f,0.8f}; int n=3;
    float *db,*ds; bool *dk;
    cudaMalloc(&db,sizeof(hb)); cudaMalloc(&ds,sizeof(hs)); cudaMalloc(&dk,n*sizeof(bool));
    cudaMemcpy(db,hb,sizeof(hb),cudaMemcpyHostToDevice);
    cudaMemcpy(ds,hs,sizeof(hs),cudaMemcpyHostToDevice);
    launch_nms(db,ds,n,0.5f,dk);
    bool hk[3]; cudaMemcpy(hk,dk,n*sizeof(bool),cudaMemcpyDeviceToHost);
    for(int i=0;i<n;i++) printf("框%d: %s\n", i, hk[i]?"保留":"抑制");
    return 0;
}
```

```bash
nvcc nms_kernel.cu test_nms.cu -o test_nms && ./test_nms
```

✅ **【检查点 D9 · 里程碑】** 输出 `框0:保留 框1:抑制 框2:保留`。**这是含金量最高的一步——你亲手写的 CUDA 核并验证正确。** 面试问"写过 CUDA 吗"，这就是答案。全错 → C5。

## Day 10 · 缓冲 + 讲 IoU/NMS 原理

🔁 **【复习 W2】** 用一句话说 NMS 在干嘛（同一物体重叠框只留分最高）。📝 为什么核函数里要 `cudaDeviceSynchronize()`？（等 GPU 算完再读结果。）

---

# 第 3 周 · C++ 调 TensorRT + 跑通内置 EfficientNMS（基线）

## Day 11-12 · C++ 调 TensorRT 推理（为阶段四铺路）

`trt_infer.cpp`（最小版，逻辑同阶段一 Python）：

```cpp
#include "NvInfer.h"
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <iostream>
using namespace nvinfer1;
class Logger : public ILogger {
    void log(Severity s, const char* m) noexcept override {
        if (s <= Severity::kWARNING) std::cout << m << std::endl;
    }
} gLogger;
int main(){
    std::ifstream f("resnet50.engine", std::ios::binary);
    std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    IRuntime* rt = createInferRuntime(gLogger);
    ICudaEngine* eng = rt->deserializeCudaEngine(d.data(), d.size());
    IExecutionContext* ctx = eng->createExecutionContext();
    const char* in = eng->getIOTensorName(0);
    const char* out = eng->getIOTensorName(1);
    ctx->setInputShape(in, Dims4{1,3,224,224});
    void *di,*doo; cudaMalloc(&di,1*3*224*224*4); cudaMalloc(&doo,1000*4);
    std::vector<float> hi(1*3*224*224,0.5f), ho(1000);
    ctx->setTensorAddress(in, di); ctx->setTensorAddress(out, doo);
    cudaStream_t st; cudaStreamCreate(&st);
    cudaMemcpyAsync(di,hi.data(),hi.size()*4,cudaMemcpyHostToDevice,st);
    ctx->enqueueV3(st);
    cudaMemcpyAsync(ho.data(),doo,ho.size()*4,cudaMemcpyDeviceToHost,st);
    cudaStreamSynchronize(st);
    int best=0; for(int i=1;i<1000;i++) if(ho[i]>ho[best]) best=i;
    std::cout << "C++ 推理预测类别: " << best << std::endl;
    cudaFree(di); cudaFree(doo); return 0;
}
```

CMakeLists：`find_package(CUDA REQUIRED)` + `include_directories(...TRT头文件...)` + `target_link_libraries(trt_infer nvinfer nvinfer_plugin cudart)`（**注意链接 `nvinfer_plugin`**，内置插件要它）。

✅ **【检查点 D11】** 编译运行打印预测类别。`undefined reference` → C3；找不到 `NvInfer.h` → C4。

## Day 13-15 · 跑通内置 EfficientNMS_TRT（关键：建立"已知正确"的基线）

> **给 AI 老师**：这步让学生用 **onnx-graphsurgeon** 把内置 `EfficientNMS_TRT` 插到 YOLOv8 ONNX 里，构建带 NMS 的引擎，得到"官方正确"的后处理输出。第 5 周自写插件时拿它对答案。参考 the0807/YOLOv8-ONNX-TensorRT 与 DeepStream-Yolo 的 end2end 做法。

要点（不逐行给，属"改仓库"，让学生跟参考仓库做）：
- `pip install onnx-graphsurgeon onnx`
- 用 graphsurgeon 在 ONNX 输出后接一个 `EfficientNMS_TRT` 节点（属性：`score_threshold`/`iou_threshold`/`max_output_boxes`）
- 构建引擎时 `initLibNvInferPlugins(&logger,"")` 注册内置插件
- 推理输出直接是 NMS 后的框（num_dets, boxes, scores, classes）

✅ **【检查点 D15】** 得到一个"内置 NMS 版"引擎，输出后处理好的框。这就是**基线**。搞不定 graphsurgeon → 先跳过，用阶段二的 Python 端 `cv2.dnn.NMSBoxes` 输出当基线也行（标 TODO）。

> **【原理速讲】** 生产里其实**直接用内置 EfficientNMS_TRT 就够了**（NVIDIA 维护、又快又稳）。我们第 5 周还要自写，纯粹是为了**学**插件机制——面试要能讲"我理解并实现过 TensorRT 插件"。先有官方正确答案，自写时才好对比。

---

# 第 4 周 · 插件"管道"（先易后难，先打通再谈性能）

> **给 AI 老师**：这周核心是搞懂插件的**生命周期和接线**，用**最简单的算子**练，把所有链接/注册错误在这里解决掉（此时核函数是平凡的，好排查）。

## Day 16-17 · 最省事的第一个插件：Python 装饰器插件（TRT 10.6+）

> TensorRT 10.6+ 提供**基于 Python 装饰器**的插件写法，比 C++ 简单得多，适合第一个"我写的插件跑起来了"的成就感。官方：https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-python.html

让学生按官方 Python 插件页做一个平凡算子（如 elementwise +1）。跑通即可。

✅ **【检查点 D17】** Python 装饰器插件在一个小网络里生效。

## Day 18-20 · C++ IPluginV3 空壳插件（打通管道）

> **给 AI 老师**：基于 **leimao/TensorRT-Custom-Plugin-Example** 改（它有完整 CMake + ONNX 集成）。**先确认它/官方页用的是 IPluginV3**；照它的结构做一个"什么都不干/只 +1"的插件，重点是把生命周期跑通。

IPluginV3 的三块（让学生对着官方页理解，别背）：
- **IPluginV3OneCore**：名字、版本、命名空间
- **IPluginV3OneBuild**：输出个数/形状/数据类型（构建期）
- **IPluginV3OneRuntime**：**`enqueue()`（在这调 CUDA 核）**、序列化

**你真正要改的核心就是 `enqueue`**（把输入输出指针拿出来，调你的核函数）：

```cpp
// enqueue 核心（IPluginV3OneRuntime）——签名以官方页为准，重点是这几行逻辑
int32_t enqueue(const PluginTensorDesc* inDesc, const PluginTensorDesc* outDesc,
                const void* const* inputs, void* const* outputs,
                void* workspace, cudaStream_t stream) noexcept override {
    const float* boxes  = static_cast<const float*>(inputs[0]);
    const float* scores = static_cast<const float*>(inputs[1]);
    bool*        keep   = static_cast<bool*>(outputs[0]);
    launch_nms(boxes, scores, mNumBoxes, mIouThr, keep);   // ← 你第2周写的核
    return 0;
}
```

CMakeLists 关键（编成 .so）：

```cmake
cmake_minimum_required(VERSION 3.18)
project(nms_plugin LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75)          # PC(WSL)=75；Jetson 改 87；两者都要:"75 87"
find_package(CUDA REQUIRED)
add_library(nms_plugin SHARED nms_plugin.cpp nms_kernel.cu)
target_link_libraries(nms_plugin nvinfer nvinfer_plugin cudart)
```

✅ **【检查点 D20】** 空壳插件编出 `.so` 并能被加载（Python `ctypes.CDLL` + `trt.init_libnvinfer_plugins` 后在 registry 里看到它）。**在这里把所有链接/注册报错解决掉**（C3/C6/C7）。

> **【原理速讲】插件部件分工**：Core=身份证；Build=告诉 TRT"我输出几个、什么形状"；Runtime.enqueue=真正干活（调核）。`REGISTER_TENSORRT_PLUGIN` 把 creator 注册进去。你真正改的就 enqueue + 输出形状，其余照参考仓库。

---

# 第 5 周 · 合体：自定义 NMS 插件 + 和基线对答案

## Day 21-23 · 把 NMS 核塞进 IPluginV3 插件

把第 2 周的 `nms_kernel.cu` 接进第 4 周的插件空壳的 `enqueue`，输出形状设为 keep 数组（长度 = 框数）。编译 `.so`。

## Day 24 · 和内置 EfficientNMS 基线对答案

同一批输入，分别跑"你的插件 NMS"和第 3 周的"内置 EfficientNMS 基线"，比较保留的框是否一致。

✅ **【检查点 D24 · 里程碑】** 你的插件输出和内置基线基本一致。**你有了一个自己做的 IPluginV3 插件 `.so`**——简历"自研 TensorRT 插件"的实物。

## Day 25 · CPU vs GPU 后处理提速对比

```python
# bench_nms.py —— CPU(cv2.dnn.NMSBoxes) vs GPU 插件
import time, numpy as np, cv2
N=5000
boxes=(np.random.rand(N,4)*640).astype(np.float32); scores=np.random.rand(N).astype(np.float32)
t=time.time()
for _ in range(100):
    xywh=np.stack([boxes[:,0],boxes[:,1],boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1]],1)
    cv2.dnn.NMSBoxes(xywh.tolist(),scores.tolist(),0.25,0.45)
print(f"CPU NMS: {(time.time()-t)/100*1000:.2f} ms/次")
# GPU 插件：跑 100 次测平均，填入对比
```

✅ **【检查点 D25】** 得到 CPU vs GPU 数字。框多时 GPU 占优；框少时 GPU 启动开销可能不划算——**"何时用 GPU 划算"本身就是面试可讲点**。

---

## 本阶段产出物（打勾）

- [ ] `cpp_basics.cpp` / `vector_add.cu`
- [ ] `nms_kernel.cu` + `test_nms.cu`（从零写、已验证）
- [ ] `trt_infer.cpp`（C++ TensorRT 推理）
- [ ] 内置 EfficientNMS 基线引擎
- [ ] `nms_plugin.cpp` + `libnms_plugin.so`（**IPluginV3**）
- [ ] `bench_nms.py`（CPU vs GPU 对比）

**面试话术**：
> "我用 CUDA 从零写过 NMS 核函数，理解 grid/block/thread 并行模型和 IoU 计算，并写测试验证正确性。"
> "我基于 TensorRT 的 **IPluginV3** 接口封装了自定义 NMS 插件，编成动态库、Python 加载。我知道生产上直接用内置 EfficientNMS_TRT 更好，自己实现是为了理解插件生命周期（Core/Build/Runtime）和 enqueue 调核的流程，并和内置插件对过答案验证正确性。"

---

## 报错对照表

**C1 · C++ 语法错** — 贴报错行号对照。缺 `;`/`#include`/大小写。

**C2 · CMake `No CMAKE_CXX_COMPILER`** — `sudo apt install build-essential`。

**C3 · `undefined reference to createInferRuntime / getPluginRegistry`** — 没链接 TRT 库。CMake 加 `target_link_libraries(... nvinfer nvinfer_plugin cudart)`，确认库路径在链接器搜索范围。

**C4 · 找不到 `NvInfer.h`** — 头文件路径没加。Jetson 在 `/usr/include/aarch64-linux-gnu/`，x86 在 `/usr/include/x86_64-linux-gnu/`；CMake `include_directories(...)` 加上。

**C5 · CUDA 核结果全错** — ① 忘 `cudaDeviceSynchronize()` 就读结果（本文档已在 launch 内同步）。② `if(idx<n)` 漏了→越界。③ `<<<grid,block>>>` 维度算错。④ `nvcc` 没设架构。贴 `.cu`。

**C6 · 编 .so `undefined reference to launch_nms`** — `nms_kernel.cu` 没进 CMake 的 `add_library` 源列表，或 `extern "C"` 声明/定义不一致。

**C7 · 插件加载了但 registry 里没有 / 反序列化时 `IPluginCreator not found`** — ① `REGISTER_TENSORRT_PLUGIN` 只能放在**一个** .cpp（别放头文件）。② 启动时要 `initLibNvInferPlugins(&logger,"")`（且链接 `nvinfer_plugin`）。③ 自定义插件的 creator 必须在 `deserializeCudaEngine` 之前注册。④ 插件名和 Python 里找的名字要一致。

**C8 · CMake `CUDA_ARCHITECTURES is empty`（CMake≥3.18）** — 必须设架构：PC(WSL) `set(CMAKE_CUDA_ARCHITECTURES 75)`，Jetson `87`，都要就 `75 87`。

**⚠️ 版本红线** — **别用 IPluginV2 写新插件**（TRT10 弃用、TRT11 移除）。用 **IPluginV3**。参考仓库若是 V2，只学结构，接口对照官方 V3 页。不确定的 V3 方法签名→查 https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-cpp.html ，别编。
