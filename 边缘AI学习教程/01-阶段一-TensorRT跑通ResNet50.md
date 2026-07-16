# 01 · 阶段一：TensorRT 跑通 ResNet50（第 1-2 周）

> **给 AI 老师**：先读 `00-教学总纲.md`。本阶段目标是**建立信心**——第一次把 PyTorch 模型变成 TensorRT 引擎跑起来、看到加速。全程纯 Python，**不装 pycuda**（用 torch 张量的显存，绕开 Windows 编译坑）。一次一小步，等学生贴结果。

**基准环境**：Windows 11 + CUDA 12.x + **TensorRT 10.x** + Quadro RTX 3000（sm_75）。

---

## 本阶段目标

```
装环境 → PyTorch 导出 ONNX → TensorRT 编译 Engine → 加载推理 → 对比速度
```

**学完能做到**：说清 ONNX / TensorRT 各干嘛；手上有能跑的 TRT 推理脚本；有一张 PyTorch vs TensorRT 速度对比表（简历第一个数据点）。

**时间**：5 个学习日（含缓冲约 2 周）。

**产出物**：`01_export_onnx.py` `02_check_onnx.py` `03_build_engine.py` `04_infer_tensorrt.py` `05_benchmark.py` `benchmark.md` + `resnet50.onnx/.engine`。

---

## 开课前 · 等板子期间做什么（Day 0，和阶段一并行）

> **给 AI 老师**：学生的 Jetson 大概第 3 天才到。**好消息：整个阶段一都在 PC 上做、完全不用板子**——正好用等货这几天跑阶段一。同时把"到货就能立刻刷机"的准备并行做好（JetPack 镜像好几个 G，别等板子到了才现下、白等几小时）。

等货这几天**并行**做（都是后台下载/看视频，不占阶段一学习时间）：
1. **硬件盘点**（学生实际配置）：
   - ✅ **已有**：Jetson Orin Nano **Super 8G** + 电源 + **NVMe SSD** + **WiFi 网卡** + IMX219 摄像头（**22pin，77°，直插 Orin Nano，不用转接线**）。
   - ✅ **不用买**：CSI 转接线（相机已 22pin）、散热风扇（Super 套件自带）、USB转TTL（Arduino 走 USB 直连 `/dev/ttyACM0`）。
   - 🛒 **阶段四要用、得买**：2 自由度云台套件（含 2 舵机+支架）、Arduino(Uno/Nano)+对应 USB 线、杜邦线、**给舵机单独的 5V 2~3A 电源**（别从板子/Arduino 取电）、小面包板(可选)。
   - 🔎 **刷机方案（已定：系统装进 SSD）**：直接刷进 NVMe SSD → **不用买 microSD**。需要一台 **Ubuntu 主机/虚拟机/Live USB** 跑 SDK Manager + 一根 **USB-C 线**。**Orin Nano 只有 DP 没 HDMI**，接显示器备 DP 线/DP转HDMI，或走无头；WiFi 天线（网卡没带就补）。
2. **提前装好刷机环境**（走 SSD 的关键）：准备一台 **Ubuntu 22.04**（虚拟机/Live USB 都行），装 **NVIDIA SDK Manager**（https://developer.nvidia.com/sdk-manager），让它把 JetPack 6.2 组件先下好（好几个 G）。被墙先问梯子。
3. **备一根 USB-C 线**：刷机时连板子进 recovery 模式用。
4. **提前看刷机视频**：B站搜 `Jetson Orin Nano 刷机 JetPack 6.2`、`Jetson CSI 摄像头 IMX219`，先看一遍，到货照做。
5. **主线**：直接开始下面的**阶段一 Day 1**（装 PC 环境）——这几天的正经学习任务。

✅ **【检查点 Day0】** 板子已下单；JetPack 镜像已下好；microSD 已烧录；刷机视频看过一遍。**板子一到（约第 3 天）直接刷，无缝接阶段二。**

> **给 AI 老师**：万一阶段一 5 天做完板子还没到，别干等——让学生复习阶段一、或提前看阶段二视频、或先啃**阶段三 Week1-2 的 C++/CUDA 基础**（也在 PC/WSL 上做，不需要板子）。总有 PC 上能干的活。

---

## 参考资源（引用时务必带【适配提示】）

| 资源 | 地址 | 用途 |
|------|------|------|
| NVIDIA TensorRT Quick Start | https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html | 官方入门，ONNX 路线 |
| 8.x→10.x Python 迁移指南 | https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-8x-to-10x-python-api.html | **查 API 新旧差异的权威** |
| ycchen218/Pytorch-to-TensorRT-example | https://github.com/ycchen218/Pytorch-to-TensorRT-example | 最干净的初学者全流程 |
| 手写AI (shouxieai) tensorRT_Pro | https://github.com/shouxieai/tensorRT_Pro | 中文最有名的部署课（B站搜"手写AI TensorRT"） |
| mmdeploy 中文教程 第六章 | https://mmdeploy.readthedocs.io/zh-cn/stable/tutorial/06_introduction_to_tensorrt.html | 免费中文图文，很干净 |

> 🔧 **【适配提示 · 通用】** 网上大量 TensorRT 教程/视频是 **TRT 8.x + x86 Linux** 录的，和你（TRT 10.x + Windows）有几处硬差异，照抄会报错。看它们**学流程**，代码以本文档为准：
> - `context.execute_v2(bindings)` / `execute_async_v2` → 你用 **`execute_async_v3`**
> - 用 binding 编号（`num_bindings`/`get_binding_name`）→ 你用**张量名**（`num_io_tensors`/`get_tensor_name`）
> - `create_network(1 << EXPLICIT_BATCH)` → 你直接 **`create_network()`**（显式 batch 已是默认，那个 flag 弃用了）
> - `config.max_workspace_size = N` → 你用 **`config.set_memory_pool_limit(...)`**
> - 很多教程用 **pycuda** 管显存 → 本教程用 **torch 张量的 `.data_ptr()`**，Windows 上省掉 pycuda 编译的坑

---

## Day 1 · 装环境 + 建立心智模型

> **给 AI 老师**：今天不写模型，只装环境——这是最容易劝退的一天，一个命令一个命令陪着来。**不要装 pycuda。**

### 步骤 1.1 — 确认显卡

```powershell
nvidia-smi
```

✅ **【检查点 D1.1】** 看到 `Quadro RTX 3000`、`6144MiB`、`CUDA Version: 12.x`。
`'nvidia-smi' 不是命令` → 【报错对照表】E1。

### 步骤 1.2 — 建独立环境

```powershell
conda create -n trt python=3.10 -y
conda activate trt
```

✅ **【检查点 D1.2】** 命令行前面出现 `(trt)`，`python --version` 是 3.10.x。

### 步骤 1.3 — 装 PyTorch（GPU 版）

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
```

✅ **【检查点 D1.3】** 打印版本 + `CUDA可用: True`。
`False` → 【报错对照表】E3。下载慢 → E2（先问梯子，别乱换源）。

### 步骤 1.4 — 装 ONNX 和 TensorRT（**不装 pycuda**）

```powershell
pip install onnx onnxruntime-gpu
pip install tensorrt
python -c "import tensorrt; print('TensorRT:', tensorrt.__version__)"
```

✅ **【检查点 D1.4 · 版本确认】** 打印 `TensorRT: 10.x.x`。**把这个版本号记住**，后面所有代码按 10.x 写。
装不上 → 【报错对照表】E4。

> **【原理速讲】整条链路在干嘛？（跑通环境后讲）**
> PyTorch/TensorFlow 像不同的"语言"，**ONNX 是通用中间格式**（翻译中转站）。**TensorRT 是针对你这张卡的优化编译器**：把模型编译成一个和显卡绑定的 `.engine`，推理时又快又省。你这两周要走一遍：PyTorch →（导出）ONNX →（编译）TensorRT engine →（加载）推理。
> **为什么不用 pycuda？** 传统教程用 pycuda 管 GPU 显存，但它在 Windows 上要装 VS 编译器、常编译失败。你已经有 PyTorch，直接用 torch 张量当显存、把它的地址喂给 TensorRT 就行——零额外依赖。

📝 **【小测 D1】** ONNX 在这条链路里扮演什么角色？（答：PyTorch 和 TensorRT 之间的通用中间格式/桥梁。）

**今日产出**：环境就绪（torch + tensorrt 都能 import，CUDA 可用）。

---

## Day 2 · PyTorch → ONNX

> **给 AI 老师**：今天让学生第一次"导出模型"。先跑通，再讲 ONNX 里存了什么。

### 步骤 2.1 — 导出脚本

`01_export_onnx.py`（完整复制）：

```python
# 01_export_onnx.py —— ResNet50 导出为 ONNX
import torch, torchvision

print("加载 ResNet50（首次下载约 100MB）...")
model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.DEFAULT).eval().cuda()

dummy = torch.randn(1, 3, 224, 224).cuda()   # 1张图, 3通道, 224x224
torch.onnx.export(
    model, dummy, "resnet50.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},  # batch 维可变
    opset_version=17,
)
print("导出完成 → resnet50.onnx")
```

```powershell
python 01_export_onnx.py
```

✅ **【检查点 D2.1】** 打印 `导出完成`，目录出现 `resnet50.onnx`（约 100MB）。报 `Unsupported operator` → E6。

### 步骤 2.2 — 验证 ONNX 正确（对比 PyTorch 输出）

`02_check_onnx.py`：

```python
# 02_check_onnx.py —— 检查 ONNX 合法 + 输出和 PyTorch 一致
import onnx, numpy as np, onnxruntime as ort, torch, torchvision

onnx.checker.check_model(onnx.load("resnet50.onnx"))   # 不合法会抛异常
print("ONNX 结构检查通过 ✅")

model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.DEFAULT).eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    ref = model(x).numpy()                              # PyTorch 输出

sess = ort.InferenceSession("resnet50.onnx", providers=["CPUExecutionProvider"])
out = sess.run(None, {"input": x.numpy()})[0]           # ONNX Runtime 输出

print("两者最大误差:", float(np.abs(ref - out).max()))
print("结果一致 ✅" if np.allclose(ref, out, atol=1e-3) else "不一致 ❌")
```

```powershell
python 02_check_onnx.py
```

✅ **【检查点 D2.2】** `ONNX 结构检查通过` + 最大误差很小（<1e-3）+ `结果一致`。

> **【原理速讲】** ONNX 文件里存两样东西：① 网络结构（计算图）；② 训练好的权重。把 `resnet50.onnx` 拖到 https://netron.app 能看到网络结构图。刚才我们用 ONNX Runtime 跑了一遍、和 PyTorch 对了答案——**验证"翻译"没翻错**，这是部署的好习惯。

🔁 **【复习 D2】** 问：昨天说 TensorRT 生成的 engine 能不能拷到另一台显卡用？（答：不能，engine 和显卡型号+TRT版本绑定。今天多记一条：ONNX 则是通用的、跨平台的。）

📝 **【小测 D2】** 为什么要用 onnxruntime 跑一遍对答案？（答：确认 PyTorch→ONNX 的导出没出错，再往下走。）

**今日产出**：`resnet50.onnx` + 验证脚本。

---

## Day 3 · ONNX → TensorRT Engine

> **给 AI 老师**：今天把 ONNX 编译成 engine。编译要几十秒到几分钟，让学生耐心等别以为死机。**注意 `create_network()` 不带 EXPLICIT_BATCH 参数（TRT10 写法）。**

### 步骤 3.1 — 构建引擎脚本

`03_build_engine.py`：

```python
# 03_build_engine.py —— ONNX 编译成 TensorRT engine（TRT 10.x 写法）
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

# TRT10：显式 batch 是默认，直接 create_network()，不要传 EXPLICIT_BATCH！
network = builder.create_network()

parser = trt.OnnxParser(network, logger)
with open("resnet50.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print("解析错误:", parser.get_error(i))
        raise SystemExit("ONNX 解析失败")

config = builder.create_builder_config()
# TRT10：用 set_memory_pool_limit 给 1GB 工作空间（旧写法 max_workspace_size 已弃用）
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# 动态 batch：告诉 TRT batch 的 最小/最优/最大
profile = builder.create_optimization_profile()
profile.set_shape("input", (1, 3, 224, 224), (1, 3, 224, 224), (8, 3, 224, 224))
config.add_optimization_profile(profile)

print("正在编译 TensorRT 引擎，请耐心等待（几十秒~几分钟）...")
serialized = builder.build_serialized_network(network, config)   # TRT10 返回序列化字节
with open("resnet50.engine", "wb") as f:
    f.write(serialized)
print("编译完成 → resnet50.engine")
```

```powershell
python 03_build_engine.py
```

✅ **【检查点 D3.1】** 中间可能有黄色 WARNING（正常），最后打印 `编译完成`，目录出现 `resnet50.engine`（几十 MB）。解析失败 → E7。

> **【原理速讲】编译期 TensorRT 干了什么？** 它不是"读了就跑"，而是针对你这张卡做优化：① **层融合**（卷积+BN+ReLU 合成一个操作，少读写显存）；② 为每层挑这张卡上最快的算法；③ 排好计算与显存复用。所以 `.engine` **和显卡绑定**——RTX 3000 上编的，拿到 Jetson 用不了，得重编。

🔁 **【复习 D3】** 问：`create_network()` 里我们没写 `EXPLICIT_BATCH`，为什么？（答：TRT10 里显式 batch 是默认，那个 flag 已弃用；这正是老教程会坑你的地方。）

**今日产出**：`resnet50.engine`。

---

## Day 4 · 加载 Engine 推理（用 torch 张量当显存，不用 pycuda）

> **给 AI 老师**：这是本阶段最"硬核"的一步，但因为用 torch 张量管显存，比传统 pycuda 版简单很多。完整给学生。

### 步骤 4.1 — 推理脚本

`04_infer_tensorrt.py`：

```python
# 04_infer_tensorrt.py —— 加载 engine 推理（TRT10 张量名 API + torch 显存）
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
d_in = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float32)  # 假输入
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
```

```powershell
python 04_infer_tensorrt.py
```

✅ **【检查点 D4.1】**
```
输入张量: input | 输出张量: output
输出 shape: (1, 1000)
预测类别编号: 某个 0~999 的数字
```
（输入是随机数，类别随机很正常；1000 = ImageNet 类别数。）报错 → E8。

> **【原理速讲】为什么用 `.data_ptr()`？** GPU 上的 torch 张量本质就是一块显存，`.data_ptr()` 是它的显存地址。TensorRT 只要知道"输入在哪、输出写到哪"两个地址就能干活。所以用 torch 张量当缓冲区，既省了 pycuda，又能和以后的前后处理（也用 torch/numpy）无缝衔接。三步不变：**准备输入张量 → execute → 读输出张量**。

🔁 **【复习 D4】** 问：TRT10 里遍历输入输出用什么？（答：`num_io_tensors` + `get_tensor_name`，按名字，不是老版的 binding 编号。）

**今日产出**：`04_infer_tensorrt.py` 能跑出预测类别。

---

## Day 5 · 速度对比（简历第一个数据）

### 步骤 5.1 — 基准测试脚本

`05_benchmark.py`：

```python
# 05_benchmark.py —— PyTorch vs TensorRT 速度对比
import time, torch, torchvision, tensorrt as trt
N = 100

# ---- A. PyTorch ----
model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.DEFAULT).eval().cuda()
x = torch.randn(1, 3, 224, 224, device="cuda")
for _ in range(10):                          # 预热
    with torch.no_grad(): model(x)
torch.cuda.synchronize()
t = time.time()
for _ in range(N):
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
for _ in range(10): run()                    # 预热
t = time.time()
for _ in range(N): run()
trt_ms = (time.time() - t) / N * 1000

print(f"PyTorch (FP32):  {pt_ms:.2f} ms/张")
print(f"TensorRT (FP32): {trt_ms:.2f} ms/张")
print(f"加速比: {pt_ms / trt_ms:.2f} 倍")
```

```powershell
python 05_benchmark.py
```

✅ **【检查点 D5.1】** 只要 TensorRT 比 PyTorch 快（加速比 > 1）就成功。RTX 3000 上大概 PyTorch ~8ms、TRT ~3ms、约 2-3 倍。**数字因机而异，不用纠结绝对值。**

> **【原理速讲】只换引擎、同样 FP32，为什么快 2-3 倍？** 全靠编译期优化（层融合、算子选优、无 Python 开销）。这就是"推理部署"岗的价值——**不改精度，纯工程手段榨速度**。面试可直接说这句。
> 想更快？编译时加一行 `config.set_flag(trt.BuilderFlag.FP16)` 用半精度，通常再快近一倍、精度几乎不掉——**第二阶段专门讲量化**。

### 步骤 5.2 — 记录成果

`benchmark.md`（简历素材）：

```markdown
# ResNet50 推理速度对比（Quadro RTX 3000, FP32）
| 框架 | 推理耗时 | 加速比 |
|------|---------|--------|
| PyTorch 原生 | X.XX ms | 1.0x |
| TensorRT | X.XX ms | X.Xx |
环境：Windows 11 + CUDA 12.x + TensorRT 10.x。结论：不损失精度下 TensorRT 加速约 X 倍。
```

🔁 **【复习 D5 · 本阶段串讲】** 让学生用一句话复述整条链路：PyTorch →(torch.onnx.export)→ ONNX →(build_serialized_network)→ engine →(execute_async_v3)→ 推理。能顺下来就过关。

**今日产出**：`05_benchmark.py` + `benchmark.md`。

---

## 本阶段产出物（打勾）

- [ ] 5 个 .py 脚本 + `resnet50.onnx/.engine` + `benchmark.md`

**面试话术**：
> "我走通过 PyTorch→ONNX→TensorRT 部署链路，在 RTX 3000 上把 ResNet50 从 X ms 优化到 Y ms、加速 Z 倍，用的是 TensorRT 10 的张量名 API 和 execute_async_v3，理解层融合和 engine 与硬件绑定的特性。"

---

## 报错对照表

**E1 · `'nvidia-smi' 不是命令`** — 驱动没装/没进 PATH。装对应 Quadro RTX 3000 的驱动后重启。

**E2 · pip 下载慢/超时** — 国内网络。先问学生**梯子**情况（别自作主张换源）。要换源用官方向导 https://pytorch.org/get-started/locally/ 选 CUDA 12.4 拿命令，确认带 `cu124`。

**E3 · `torch.cuda.is_available()` = False** — 装成 CPU 版。`pip uninstall torch torchvision -y` 后按 D1.3 重装；确认 `nvidia-smi` 的 CUDA ≥ 12.4。

**E4 · `pip install tensorrt` 失败** — 让学生贴完整报错。常见：Python 版本不在 3.8-3.12。备选加 `--extra-index-url https://pypi.nvidia.com`，或按 CUDA 版装 `pip install tensorrt-cu12`。拿不准就一起查官方安装文档 https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html。

**E6 · 导出报 `Unsupported operator`** — ResNet50 正常不会遇到。真遇到贴 `torch.__version__`，把 `opset_version=17` 改成 13 试。

**E7 · 解析 ONNX 失败** — 脚本已打印 `解析错误`，让学生贴出来。常见是 opset 太新，降 opset 重导（D2.1 里 17 改小）。

**E8 · 推理报错** — ① 张量名对不对（先跑 D4 打印的名字）。② 6GB 跑 ResNet50 batch=1 绰绰有余，若显存不足，关掉浏览器/别的 python 进程，`nvidia-smi` 看谁在占。③ `AttributeError: execute_async_v3` → 你的 TensorRT 是 8.x（见下）。

**⚠️ TRT8 差异**（`import tensorrt` 版本是 8.x 时）：`execute_async_v3`→`execute_async_v2(bindings, stream)`（bindings 是 `[d_in.data_ptr(), d_out.data_ptr()]`）；`set_tensor_address`/`get_tensor_name` 这套张量名 API 换成 binding 编号 API；`create_network()` 要写回 `create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))`。**建议直接升级到 TRT10 免这些麻烦。**
