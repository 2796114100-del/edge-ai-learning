# 05 · 阶段五：端侧大模型 llama.cpp（第 16-18 周）

> **给 AI 老师**：先读 `00-教学总纲.md`。本阶段**加分项**，Jetson 上离线跑小型大模型，理解端侧 LLM 部署链路。代码少命令多，是硬仗后的调剂。原则：**不改源码，只编译/量化/运行/测性能**。**转换和量化在 PC 做（Jetson 装 PyTorch 麻烦），推理在 Jetson 做。**

**基准环境**：Jetson Orin Nano 8GB + JetPack 6.x（CUDA 12.6，sm_87）+ `ggml-org/llama.cpp`（**CMake 构建**）+ Qwen2.5-0.5B。

> ⚠️ **给 AI 老师三条铁提醒**：
> 1. 仓库是 **`ggml-org/llama.cpp`**（旧 `ggerganov` 会重定向）。构建用 **CMake**：**老教程的 `make LLAMA_CUDA=1` 已废弃**（见 J1 / 适配提示）。
> 2. Jetson 编译用 **`-DCMAKE_CUDA_ARCHITECTURES=87`** + **`-j4`**（不然 8GB 内存编译容易 OOM）。
> 3. **转换/量化在 PC 上做**（纯 CPU、跨平台），把 `.gguf` `scp` 到 Jetson 只跑推理。

---

## 本阶段目标

```
Jetson 编译 llama.cpp(CUDA) → (PC上)下载模型→转GGUF→量化 → scp 到 Jetson
→ 跑对话 → llama-bench 测速 → 三量化对比报告
```

**学完能做到**：边缘设备离线跑大模型；理解 GGUF 和 Q4/Q8 量化；有一份端侧 LLM 量化性能报告（第二个量化数据点）。

**时间**：约 3 周（轻松）。

---

## 参考资源（引用务必带【适配提示】）

| 资源 | 地址 | 用途 |
|------|------|------|
| llama.cpp 官方 build 文档 | https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md | **权威构建命令** |
| Qwen 官方 llama.cpp 量化指南 | https://qwen.readthedocs.io/en/v2.5/quantization/llama.cpp.html | 转换/量化官方步骤 |
| NVIDIA Jetson AI Lab | https://github.com/NVIDIA-AI-IOT/jetson-ai-lab | 官方边缘 LLM 教程 |
| kreier gist（Jetson+CUDA） | https://gist.github.com/kreier/6871691130ec3ab907dd2815f9313c5d | Jetson 编译实操 |
| 预制 GGUF（跳过转换） | https://huggingface.co/AmpereComputing/qwen-2.5-0.5b-instruct-gguf | Day1 想直接跑就用它 |

> 🔧 **【适配提示 · 通用】** llama.cpp 更新极快，**2023-2024 的教程几乎都过时**：① 构建从 `make LLAMA_CUDA=1` 变成了 `cmake -B build -DGGML_CUDA=ON`；② 可执行文件从 `./main` 改名成 `./build/bin/llama-cli`；③ 转换脚本是 `convert_hf_to_gguf.py`（不是老的 `convert.py`）。看老视频学**思路**，命令以本文档/官方 build.md 为准。

---

# 第 1 周 · 环境 + 编译 + 首次运行

## Day 1-2 · 刷机就绪 + 加 swap + 开 Super

确认 JetPack 6.x（`nvcc --version` = CUDA 12.6）。开 Super：`sudo nvpmodel -m 0 && sudo jetson_clocks`。**加 8GB swap**（编译防 OOM）：
```bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```
✅ **【检查点 D2】** `free -h` 看到 8G swap；`nvcc` 是 12.6。

## Day 3 · 克隆 + 读 build 文档

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
```
（被墙先问梯子：`export HTTPS_PROXY=http://127.0.0.1:7890`。）读 `docs/build.md`，理解为什么 CMake 取代了 Makefile。

## Day 4 · 编译（带 CUDA，指定架构）

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build --config Release -j4        # -j4 防 8GB 内存编译 OOM
./build/bin/llama-cli --version
```
✅ **【检查点 D4】** 打印版本，`build/bin/` 下有 `llama-cli` `llama-quantize` `llama-bench`。找不到 CUDA → J1；编译中 OOM → J2。

> **【原理速讲】llama.cpp 为什么适合端侧？** 纯 C++、极轻量，不依赖 PyTorch 那一大坨；配 GGUF 量化，几亿参数模型能压到几百 MB，小设备也能离线跑。`GGML_CUDA=ON` 让它用 GPU；`sm_87` 是 Orin 的架构。

## Day 5-7 · 先跑一个预制 GGUF（快速成就感）+ 学 CLI

```bash
# 直接下一个现成的 GGUF，先跑起来（不用等转换）
# 从 AmpereComputing/qwen-2.5-0.5b-instruct-gguf 下一个 q4 文件，然后：
./build/bin/llama-cli -m qwen2.5-0.5b-q4.gguf -ngl 99 -c 2048 -cnv -p "你好，介绍一下你自己"
```
✅ **【检查点 D5】** 模型输出中文回答；`jtop` 看到 GPU 有负载。乱码/崩溃 → J3。
学 `-ngl 99`（所有层放 GPU）、`-cnv`（对话模式）、`-c 2048`（上下文长度）。

> **Week1 里程碑**：**Jetson 离线跑起了大模型**，录个视频（断网状态更有说服力）。

---

# 第 2 周 · 自己搭转换/量化流水线（在 PC 上做）

## Day 8-9 · 下载模型 + 转 GGUF

在 **PC**（RTX 3000，装 PyTorch 方便）上、llama.cpp 目录里：
```bash
python -m venv venv && venv\Scripts\activate      # Windows；Linux: source venv/bin/activate
pip install -r requirements.txt                    # 转换脚本依赖
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct   # 被墙走梯子或用 modelscope
python convert_hf_to_gguf.py ./Qwen2.5-0.5B-Instruct --outfile qwen-f16.gguf --outtype f16
```
✅ **【检查点 D9】** 生成 `qwen-f16.gguf`（约 1GB）。报错 → J4。

## Day 10 · 量化成 Q8_0 / Q4_K_M / Q4_0

```bash
./build/bin/llama-quantize qwen-f16.gguf qwen-q8_0.gguf  Q8_0
./build/bin/llama-quantize qwen-f16.gguf qwen-q4_k_m.gguf Q4_K_M   # 推荐：低比特里质量好
./build/bin/llama-quantize qwen-f16.gguf qwen-q4_0.gguf  Q4_0
ls -lh *.gguf                                       # 比大小：Q4 < Q8 < f16
```

> **【原理速讲】GGUF 量化**：和阶段二 INT8 同思路——更少的位表示权重来省空间。F16(原始)＞Q8_0(≈半，几乎无损)＞Q4_K_M/Q4_0(≈1/4，质量略降)。**Q4_K_M 通常比 Q4_0 质量好**（K 系列是改进量化），低显存优先它。

📝 **【小测 D10】** 端侧 LLM 部署的核心矛盾是什么？（答：模型质量 vs 显存/速度，量化是这个权衡的旋钮。）

## Day 11-14 · scp 到 Jetson + llama-bench 实测

把三个 `.gguf` `scp` 到 Jetson，逐个 `llama-bench`：
```bash
./build/bin/llama-bench -m qwen-q4_k_m.gguf -ngl 99
```
记录 tok/s + 显存（`jtop`）。同一 prompt 跑三量化比质量。

> **【原理速讲】** 生成速度是**显存带宽瓶颈**。Orin Nano 带宽 68GB/s（Super 模式 102GB/s）。0.5B 模型很小，8GB 绰绰有余，OOM 不是问题（到 3B+ 才需操心）。**tok/s 必须自己 `llama-bench` 实测**，别抄网上数字。

---

# 第 3 周 · 应用 + 收尾

## Day 15-16 · llama-server + Python 客户端

```bash
./build/bin/llama-server -m qwen-q4_k_m.gguf -ngl 99 -c 2048
# 另一终端用 curl / python 调 OpenAI 兼容接口 /v1/chat/completions
```

## Day 17 · 功耗研究

`tegrastats`/`jtop` 对比 15W 模式 vs Super 模式的 tok/s 和每瓦性能。

## Day 18 · 试更大模型感受 8GB 天花板

跑 Qwen2.5-1.5B/3B，体会显存压力；学**运行期救急标志**（大模型才需要）：
```bash
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ./build/bin/llama-server -m 大模型.gguf -ngl 99
```

## Day 19-21 · 性能报告 + Demo

`edge_llm_report.md`：
```markdown
# Qwen2.5-0.5B 端侧量化对比（Jetson Orin Nano 8GB）
| 量化 | 文件大小 | 显存 | tok/s(实测) | 质量(主观) |
|------|---------|------|------------|-----------|
| F16  | ~1.0GB | XX | XX | 基准 |
| Q8_0 | ~0.5GB | XX | XX | 几乎无损 |
| Q4_K_M | ~0.4GB | XX | XX | 略降但可用 |
结论：Q4_K_M 显存/速度最优，质量敏感用 Q8_0。CUDA offload(-ngl 99) 对速度提升明显。
```
录一段离线对话 Demo 视频。

---

## 本阶段产出物（打勾）

- [ ] 编译好的 llama.cpp（带 CUDA）
- [ ] `qwen-f16/q8_0/q4_k_m.gguf`
- [ ] `edge_llm_report.md`（量化对比）
- [ ] 离线大模型对话 Demo 视频

**面试话术**：
> "我在 Jetson Orin Nano 上用 llama.cpp（CMake + GGML_CUDA）部署 Qwen2.5-0.5B，做了 F16/Q8_0/Q4_K_M 量化对比、用 llama-bench 实测 tok/s。理解 GGUF 格式、端侧质量与资源的权衡，也知道 3B+ 模型要靠 unified memory 应对 8GB 显存上限。"

---

## 报错对照表

**J1 · 编译找不到 CUDA/nvcc** — ① `nvcc --version` 确认在 PATH。② 加 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`。③ 先去掉 `-DGGML_CUDA=ON` 用 CPU 编译跑通（0.5B CPU 也能跑，只是慢），GPU 当优化项后补。

**J2 · 编译中 OOM（进程被杀）** — 8GB 内存吃紧。① 一定加 `-DCMAKE_CUDA_ARCHITECTURES=87`（少编好几个架构）。② `-j4` 或 `-j2` 降并行。③ 已加 swap（Day2）。④ 在 `tmux` 里编，SSH 断了也不中断。

**J3 · 推理乱码/崩溃** — ① 中文乱码多是终端编码，不影响模型。② 显存不足→换更小量化或减 `-ngl`。③ gguf 转坏→重转。④ JetPack6 有个驱动 bug 报 `unable to allocate CUDA0 buffer`，查 JetPack 补丁级别/NVIDIA 论坛，别当成编译错。

**J4 · `convert_hf_to_gguf.py` 报错** — ① 缺依赖→`pip install -r requirements.txt`。② 模型文件不全→确认 safetensors+config 都下全（git-lfs）。③ 脚本名因版本不同→`ls convert*.py` 看实际名。

**J5 · 下载模型极慢/连不上** — 先问学生梯子（别自作主张换源）。开梯子 `export HTTPS_PROXY=http://127.0.0.1:7890`，或用 modelscope 国内源下同名模型，或直接用预制 GGUF（Day5）。

**⚠️ 版本红线** — 别教 `make LLAMA_CUDA=1`（废弃）或 `./main`（改名了）。用 `cmake -B build -DGGML_CUDA=ON` + `./build/bin/llama-cli`。
