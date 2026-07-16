# 02 · 阶段二：Jetson 部署 YOLOv8 + 量化（第 3-6 周）

> **给 AI 老师**：先读 `00-教学总纲.md`。这是**整个计划的核心项目**，简历主力。策略：**先用 Ultralytics 一行命令拿到"跑起来了"的成就感，再拆开引擎盖手写前后处理 + 理解 INT8**。量化是面试重点（第 3 周教扎实）。硬件（刷机/摄像头）给"视频 + 验证命令 + 适配提示"。一次一小步，等结果。

**基准环境**：Jetson Orin Nano 8GB + **JetPack 6.2.x**（Ubuntu 22.04 / CUDA 12.6 / **TensorRT 10.3** / sm_87）+ **YOLOv8**（Ultralytics）。

> ⚠️ **给 AI 老师的两条铁提醒（本阶段最常翻车点）**：
> 1. **引擎和设备绑定**——所有 `.engine` 必须**在 Jetson 上**生成，不能从 PC 拷。
> 2. **pip 装 opencv/ultralytics 会悄悄顶掉 JetPack 自带的、带 GStreamer 的 OpenCV，导致 CSI 摄像头打不开**。用独立 venv 隔离，摄像头相关代码用系统 OpenCV。见 Day 4 / 报错表 J2。

---

## 本阶段目标

```
刷 JetPack 6.2 → 点亮 CSI 摄像头 → YOLOv8 一行跑通(Ultralytics) → 导出 engine(FP16)
→ 拆引擎盖：手写前处理+后处理+推理 → INT8 量化+校准 → 三精度对比 → 部署报告
```

**学完能做到**：Jetson 从零到实时检测；手写 YOLOv8 的 letterbox 前处理和 `[1,84,8400]` 解码+NMS；做出 FP32/FP16/INT8 对比；讲清 INT8 校准（面试重点）。

**时间**：约 4 周（~20 天）。

---

## 参考资源（引用务必带【适配提示】）

| 资源 | 地址 | 用途 |
|------|------|------|
| Ultralytics NVIDIA Jetson 指南 | https://docs.ultralytics.com/guides/nvidia-jetson/ | **官方权威**：Jetson 安装 + 导出 + 基准 |
| dusty-nv/jetson-inference | https://github.com/dusty-nv/jetson-inference | 绝对新手的 Day1 信心（Hello AI World） |
| triple-Mu/YOLOv8-TensorRT | https://github.com/triple-Mu/YOLOv8-TensorRT | Python 端到端 + CUDA 后处理，看"中间层" |
| the0807/YOLOv8-ONNX-TensorRT | https://github.com/the0807/YOLOv8-ONNX-TensorRT | FP16+INT8 摄像头 demo 并排对比 |
| marcoslucianops/DeepStream-Yolo | https://github.com/marcoslucianops/DeepStream-Yolo | 生产级 + 有清晰的 INT8 校准文档 |
| 手写AI (shouxieai) | https://github.com/shouxieai/infer | 中文部署教育（B站搜"手写AI TensorRT 部署"） |

> 🔧 **【适配提示 · 通用】** B 站/CSDN 上大量"YOLO + Jetson + TensorRT"教程是 **YOLOv5** 或 **老 JetPack(TRT8)** 或 **x86** 录的：
> - **YOLOv5 输出 `[1,25200,85]`（含 objectness）** ↔ 你的 **YOLOv8 是 `[1,84,8400]`（无 objectness，多一次 Transpose）**——后处理代码不能照抄，逻辑见 Day 8。
> - 老教程 `make`/`execute_v2`/`EXPLICIT_BATCH` 都是 TRT8 写法，你按本文档 TRT10 写。
> - x86 教程在 PC 上导引擎；你**必须在 Jetson 上导**。
> 看它们学**流程和思路**，代码以本文档为准。

---

# 第 1 周 · 环境 + 用 Ultralytics 快速跑通

## Day 1 · 刷机 + 验证 + 开 Super Mode

> **给 AI 老师**：纯硬件操作，跟视频做。B站搜 `Jetson Orin Nano 刷机 JetPack 6.2`。你的活是核对验证命令输出。
>
> 🕐 **先确认板子到了吗？** 等货期间学生应已在跑阶段一(PC)、并按阶段一顶部"Day 0 清单"提前下好了 JetPack 镜像。**板子没到就别开这一阶段**——继续阶段一，或先啃阶段三 Week1-2 的 C++/CUDA 基础（也在 PC 上做）。到货了再回来刷机。有提前下好的镜像，刷机会快很多。

**刷机（系统装进 SSD，用 SDK Manager）**——学生已选 SSD 启动：
1. 一台 **Ubuntu 22.04 主机/虚拟机/Live USB** 装 **NVIDIA SDK Manager**。
2. USB-C 连板子和主机，Jetson 进 **Force Recovery 模式**（按住 REC 键再上电，具体看套件说明）；主机 `lsusb` 能看到 NVIDIA 设备即对。
3. SDK Manager 选 JetPack **6.2** / 目标 Orin Nano / **存储选 NVMe SSD**，开刷（系统+CUDA+TensorRT 都进 SSD）。
4. 刷完从 SSD 首次启动，做 Ubuntu 初始配置（接 DP 显示器或走 USB-C 无头串口）。
> ⚠️ VM 的 USB 直通在 recovery 那步偶尔不稳，不行就用 **Ubuntu Live USB** 更省心；新板子若提示先更新 QSPI 固件，SDK Manager 会一并处理。

首次开机进桌面后：

```bash
# 版本确认（本阶段所有代码按这些版本写）
nvcc --version                                   # 期望 CUDA 12.6
python3 -c "import tensorrt; print(tensorrt.__version__)"   # 期望 10.3.x

# 开最高性能 + Super Mode（Orin Nano 8GB 关键，算力从 40→67 INT8 TOPS）
sudo nvpmodel -m 0
sudo jetson_clocks

# 装监控工具
sudo pip3 install jetson-stats
sudo jtop        # 看 CPU/GPU/内存/温度；按 q 退出
```

✅ **【检查点 D1 · 版本确认】** CUDA 12.6、TensorRT 10.3.x、`jtop` 界面正常。报错 → J1。

**今日产出**：Jetson 就绪，版本记录在案。

## Day 2 · Ultralytics 一行跑通（Day1 级成就感）

> 先用独立 venv，避免污染系统 Python（尤其别顶掉系统 OpenCV，见 Day 4）。

```bash
python3 -m venv ~/yolo_env && source ~/yolo_env/bin/activate
pip install ultralytics
# 对一张图跑检测（首次会下载 yolov8n.pt）
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

✅ **【检查点 D2】** 终端打印检测到的目标，`runs/detect/predict/` 下生成画好框的结果图。

> **【原理速讲】** `yolov8n` 的 `n`=nano，最小最快，最适合边缘设备。Ultralytics 把前处理、推理、后处理全包了——**今天先享受"它替你干完"，第 2 周我们再把这些手写一遍，才是面试要讲的东西。**

📝 **【小测 D2】** 为什么边缘设备优先选 yolov8n 而不是 yolov8x？（答：n 参数少、快、省显存，边缘要实时。）

**今日产出**：第一张检测结果图。

## Day 3 · 检测基础概念 + 视频检测

```bash
yolo predict model=yolov8n.pt source='你的一段.mp4' show=True
```

> **【原理速讲】** 讲清 5 个词（面试基础）：**bbox**（框 x1y1x2y2）、**class**（类别）、**confidence**（置信度）、**IoU**（交并比=两框重叠程度）、**NMS**（非极大值抑制：同一物体的重叠框只留分最高的）。这些第 2 周手写后处理会全用到。

🔁 **【复习 D3】** 回顾阶段一：engine 能跨设备复用吗？（不能）→ 引出今天：所以我们等下导出 engine 要在 Jetson 上导。

**今日产出**：能跑视频检测，理解检测基本概念。

## Day 4 · 点亮 CSI 摄像头（本周最易翻车）

> ⚠️ **给 AI 老师**：**pip 装的 opencv-python 没有 GStreamer 支持**，会顶掉 JetPack 自带的带 GStreamer 的 OpenCV，导致 `nvarguscamerasrc` 打不开。处理见下 + J2。插排线**断电**、看金手指朝向。

```bash
# 1. 确认摄像头识别
ls /dev/video*                    # 期望有 /dev/video0
# 2. gstreamer 预览（IMX219）
gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! xvimagesink
```

用 OpenCV 打开 CSI（存成 `cam_test.py`，**用系统 python3 跑，别在 pip-opencv 的 venv 里跑**）：

```python
# cam_test.py —— 用系统 OpenCV(带GStreamer) 打开 CSI 摄像头
import cv2
def gst_pipeline(w=1280, h=720, fps=30, flip=0):
    return (f"nvarguscamerasrc ! video/x-raw(memory:NVMM),width={w},height={h},"
            f"framerate={fps}/1 ! nvvidconv flip-method={flip} ! "
            f"video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true")
cap = cv2.VideoCapture(gst_pipeline(), cv2.CAP_GSTREAMER)
print("摄像头打开:", cap.isOpened())
ok, frame = cap.read()
if ok:
    cv2.imwrite("cam_shot.jpg", frame); print("拍到一帧，存 cam_shot.jpg")
cap.release()
```

```bash
python3 cam_test.py          # 系统 python3（自带带 GStreamer 的 OpenCV）
```

✅ **【检查点 D4】** `摄像头打开: True` + 生成 `cam_shot.jpg`。`False`/报错 → J2。

> **【原理速讲】** CSI 摄像头走 NVIDIA 的 `nvarguscamerasrc`（不是普通 USB 摄像头的 V4L2），必须用 GStreamer 管道喂给 OpenCV。这就是为什么必须用**带 GStreamer 编译的 OpenCV**（JetPack 自带的那个）。

🔧 **【适配提示】** 网上 Jetson 摄像头教程若用 `cv2.VideoCapture(0)` 直接开——那是 USB 摄像头写法，你的 CSI 要用上面的 GStreamer 管道字符串。

**今日产出**：`cam_test.py` 能拍到一帧。

## Day 5 · 缓冲 / 整理 / 记笔记

回顾本周，把踩的坑记进 `deploy_report.md` 的"踩坑记录"。缓冲没做完的。

---

# 第 2 周 · 导出 engine + 拆引擎盖（手写前后处理）

## Day 6 · 导出 TensorRT engine（FP16）

> **在 Jetson 上导。** Ultralytics 一行搞定：

```bash
# FP16 引擎（half=True）——在 Jetson 上运行
yolo export model=yolov8n.pt format=engine half=True imgsz=640
# 用引擎推理，明显更快
yolo predict model=yolov8n.engine source='https://ultralytics.com/images/bus.jpg'
```

✅ **【检查点 D6】** 生成 `yolov8n.engine`；用它 predict 正常出框。

> **【原理速讲】FP16 为什么"白捡的加速"？** FP16 是 16 位浮点，数据量减半、Jetson 有半精度计算单元、网络对这点精度损失不敏感——所以几乎免费提速。部署几乎默认开 FP16。

**今日产出**：`yolov8n.engine`（FP16）。

## Day 7 · 导出 ONNX + 看懂 YOLOv8 的输出

```bash
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12
```

把 `yolov8n.onnx` 拖到 https://netron.app 看输出：**`[1, 84, 8400]`**。

> **【原理速讲 · 关键，面试常问】** 记住这两组数：
> - **8400** = 三个尺度的候选框总数（80×80+40×40+20×20）。
> - **84** = 4 个框坐标 + 80 个类别分。**注意：YOLOv8 没有 objectness 那一列**（YOLOv5 的 85 = 4+1+80，多的那个 1 就是 objectness）。
> - YOLOv8 是 **anchor-free**（不预设锚框），YOLOv5 是 anchor-based。
> 所以 YOLOv8 后处理和 YOLOv5 **不一样**，别照抄 v5 教程。

🔁 **【复习 D7】** 阶段一 ONNX 里存什么？（结构+权重）→ 今天用 netron 亲眼看。

📝 **【小测 D7】** YOLOv8 输出 84 列里为什么没有 objectness？（答：v8 是 anchor-free 解耦头，直接用类别分当置信度。）

**今日产出**：`yolov8n.onnx` + 看懂输出维度。

## Day 8 · 手写前处理 + 后处理（本阶段技术核心）

> **给 AI 老师**：完整给学生，逐段讲。让学生建 `yolo_common.py`，后面推理/量化都 import 它。**这是 YOLOv8 版（84/8400、无 objectness、要 Transpose）。**

```python
# yolo_common.py —— YOLOv8 前处理/后处理/画框
import cv2, numpy as np

CLASS_NAMES = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse',
'sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard',
'surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana',
'apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
'hair drier','toothbrush']

def letterbox(img, new_shape=640, color=(114,114,114)):
    """等比缩放+填灰边到 640x640。返回 处理后图, 缩放比 r, 填充(dw,dh)"""
    h, w = img.shape[:2]
    r = min(new_shape/h, new_shape/w)
    nw, nh = int(round(w*r)), int(round(h*r))
    dw, dh = (new_shape-nw)/2, (new_shape-nh)/2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, (dw, dh)

def preprocess(img):
    """图 → (1,3,640,640) float32, 0~1, RGB, CHW"""
    padded, r, (dw, dh) = letterbox(img)
    blob = padded[:, :, ::-1].transpose(2,0,1)          # BGR→RGB, HWC→CHW
    blob = np.ascontiguousarray(blob, dtype=np.float32)/255.0
    return blob[np.newaxis, ...], r, (dw, dh)

def postprocess(output, r, dw_dh, conf_thres=0.25, iou_thres=0.45):
    """YOLOv8 输出 (1,84,8400) → 检测框（坐标已还原到原图）"""
    dw, dh = dw_dh
    pred = output[0].T                    # (8400, 84)  —— 关键：转置！
    boxes = pred[:, :4]                   # cx,cy,w,h
    cls_scores = pred[:, 4:]              # 80 类（无 objectness）
    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(len(pred)), class_ids]
    keep = scores > conf_thres
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
    if len(scores) == 0:
        return []
    cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x1 = (cx - w/2 - dw)/r;  y1 = (cy - h/2 - dh)/r      # 还原 letterbox
    x2 = (cx + w/2 - dw)/r;  y2 = (cy + h/2 - dh)/r
    boxes_nms = np.stack([x1, y1, x2-x1, y2-y1], axis=1)  # NMS 要 xywh
    idxs = cv2.dnn.NMSBoxes(boxes_nms.tolist(), scores.tolist(), conf_thres, iou_thres)
    out = []
    for i in np.array(idxs).flatten():
        out.append((int(x1[i]),int(y1[i]),int(x2[i]),int(y2[i]),float(scores[i]),int(class_ids[i])))
    return out

def draw(img, dets):
    for x1,y1,x2,y2,s,c in dets:
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img,f"{CLASS_NAMES[c]} {s:.2f}",(x1,max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    return img
```

> **【原理速讲】** ① **letterbox**：直接 resize 会把物体压扁、框歪；letterbox 等比缩放+填灰边保持比例，后处理再把坐标"减填充、除缩放比"还原回原图。② **`.T` 转置**：YOLOv8 输出是 `(84, 8400)`，要转成 `(8400, 84)` 才好按"每个框一行"处理——这一步就是和 YOLOv5 的关键差异。③ **NMS**：去掉同一物体的重叠框。

📝 **【小测 D8】** 后处理里 `pred[0].T` 为什么要转置？（答：v8 输出把 84 放在前面，转成每行一个框才好算。）

**今日产出**：`yolo_common.py`。

## Day 9 · 手写 TensorRT 推理（单图，用 pycuda）

> **给 AI 老师**：Jetson 上 **pycuda 编译无障碍**（Windows 才有那个坑），所以这里用 pycuda 管显存。`pip install pycuda`。

```python
# infer_image.py —— 手写 TensorRT 推理（对一张图）
import cv2, numpy as np, tensorrt as trt
import pycuda.driver as cuda, pycuda.autoinit
from yolo_common import preprocess, postprocess, draw, CLASS_NAMES

ENGINE, IMAGE = "yolov8n.engine", "cam_shot.jpg"     # 用 Day6 的引擎 + Day4 拍的图
logger = trt.Logger(trt.Logger.WARNING)
with open(ENGINE, "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
in_name, out_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
context.set_input_shape(in_name, (1,3,640,640))
out_shape = tuple(context.get_tensor_shape(out_name))

img = cv2.imread(IMAGE)
blob, r, dw_dh = preprocess(img)
blob = np.ascontiguousarray(blob)
out_host = np.empty(out_shape, dtype=np.float32)
d_in = cuda.mem_alloc(blob.nbytes)
d_out = cuda.mem_alloc(out_host.nbytes)
context.set_tensor_address(in_name, int(d_in))
context.set_tensor_address(out_name, int(d_out))
stream = cuda.Stream()

cuda.memcpy_htod_async(d_in, blob, stream)
context.execute_async_v3(stream_handle=stream.handle)
cuda.memcpy_dtoh_async(out_host, d_out, stream)
stream.synchronize()

dets = postprocess(out_host, r, dw_dh)
cv2.imwrite("result.jpg", draw(img, dets))
print(f"检测到 {len(dets)} 个目标 → result.jpg")
for x1,y1,x2,y2,s,c in dets: print(f"  {CLASS_NAMES[c]}: {s:.2f}")
```

```bash
python3 infer_image.py
```

✅ **【检查点 D9 · 里程碑】** 打印检测目标，`result.jpg` 框画得准。框全歪 → J4。
**这是第一个里程碑**：你手写的前后处理 + 原生 TensorRT 推理，结果和 Ultralytics 一致。截图存档。

🔁 **【复习 D9】** 阶段一用 torch `.data_ptr()`，今天用 pycuda `mem_alloc` —— 都是"给 TensorRT 一个显存地址"，只是工具不同（Jetson 上 pycuda 好用）。

**今日产出**：`infer_image.py` 手写推理跑通。

## Day 10 · 摄像头实时检测

```python
# infer_camera.py —— CSI 实时检测（复用 infer_image 的推理逻辑）
import cv2, time, numpy as np, tensorrt as trt
import pycuda.driver as cuda, pycuda.autoinit
from yolo_common import preprocess, postprocess, draw

logger = trt.Logger(trt.Logger.WARNING)
with open("yolov8n.engine","rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()
in_name, out_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
ctx.set_input_shape(in_name,(1,3,640,640))
out_host = np.empty(tuple(ctx.get_tensor_shape(out_name)), dtype=np.float32)
d_in=d_out=None; stream=cuda.Stream()

def infer(img):
    global d_in,d_out
    blob,r,dw_dh = preprocess(img); blob=np.ascontiguousarray(blob)
    if d_in is None:
        d_in=cuda.mem_alloc(blob.nbytes); d_out=cuda.mem_alloc(out_host.nbytes)
        ctx.set_tensor_address(in_name,int(d_in)); ctx.set_tensor_address(out_name,int(d_out))
    cuda.memcpy_htod_async(d_in,blob,stream)
    ctx.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(out_host,d_out,stream); stream.synchronize()
    return postprocess(out_host,r,dw_dh)

def gst(w=1280,h=720,fps=30,flip=0):
    return (f"nvarguscamerasrc ! video/x-raw(memory:NVMM),width={w},height={h},"
            f"framerate={fps}/1 ! nvvidconv flip-method={flip} ! video/x-raw,format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true")
cap = cv2.VideoCapture(gst(), cv2.CAP_GSTREAMER)
while True:
    ok, frame = cap.read()
    if not ok: break
    t=time.time(); dets=infer(frame); fps=1/(time.time()-t)
    frame=draw(frame,dets)
    cv2.putText(frame,f"FPS:{fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow("YOLOv8",frame)
    if cv2.waitKey(1)==ord('q'): break
cap.release(); cv2.destroyAllWindows()
```

✅ **【检查点 D10 · 里程碑】** 实时画面，左上 FPS，物体被框出。Orin Nano(Super) 上 yolov8n FP16 端到端约 **30-60 FPS**。低于 10 → J5。**录视频存档，简历要用。**

**今日产出**：`infer_camera.py` 实时检测 Demo。

---

# 第 3 周 · INT8 量化（面试差异化重点）

## Day 11 · 量化原理（动手前先讲清）

> **【原理速讲 · 面试高频】**
> - **FP32→FP16**：精度砍半、范围够用，直接转，几乎不掉点。
> - **FP32→INT8**：INT8 只有 256 个整数格子。要把浮点塞进去需要**缩放系数 scale**。每层激活值范围不同，scale 定多少最好？
> - **校准(calibration)**：拿一批**有代表性的真实图**跑一遍，统计每层激活分布，为每层算最佳 scale。方法有 **MinMax** 和 **Entropy**（YOLO 常用 Entropy，精度通常更好）。
> - **PTQ vs QAT**：上面这套是 **PTQ**（训练后量化），不重训、最常用。PTQ 掉点太狠才上 **QAT**（训练时模拟量化，精度更好但要重训）。**面试记：先 PTQ，不够再 QAT。**
> - **代价**：INT8 通常比 FP32 快 1.5-3 倍、显存降不少，精度掉约 3-7%（要实测，别瞎说）。

📝 **【小测 D11】** 为什么 FP16 不用校准、INT8 要？（答：FP16 范围够；INT8 只有 256 格，必须靠校准集统计每层 scale。）

## Day 12 · 准备校准集 + 一行 INT8

```bash
# Ultralytics 一行 INT8（底层就是 entropy 校准，data 指向数据集配置）
yolo export model=yolov8n.pt format=engine int8=True data=coco8.yaml imgsz=640
```

> 校准集要"像真实使用场景"，官方建议 **≥1000 张有代表性的图**。教学先用 `coco8`（自带小样例）跑通流程，再换 ≥1000 张自己的场景图重导。

✅ **【检查点 D12】** 生成 INT8 引擎；用它 predict，框基本还准（可能个别小目标漏检）。

## Day 13 · 拆引擎盖：手写 INT8 校准器

> **给 AI 老师**：这是"理解 INT8 底层"的关键代码。**校准器的前处理必须和推理时完全一致**（都用 `yolo_common` 的 letterbox+/255），否则校准出来的 scale 是错的——这是最常见的坑（Day 15 专门 debug）。

```python
# build_int8.py —— 手写 entropy 校准器构建 INT8 引擎
import os, glob, numpy as np, cv2, tensorrt as trt
import pycuda.driver as cuda, pycuda.autoinit
from yolo_common import letterbox

ONNX, ENGINE, CALIB_DIR, CACHE = "yolov8n.onnx","yolov8n_int8.engine","calib_images","calib.cache"

class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_dir, cache):
        super().__init__()
        self.cache = cache
        self.imgs = glob.glob(os.path.join(calib_dir,"*.jpg"))
        self.idx = 0
        self.d_in = cuda.mem_alloc(1*3*640*640*4)          # 一张 float32 图
    def get_batch_size(self): return 1
    def get_batch(self, names):
        if self.idx >= len(self.imgs): return None          # 喂完返回 None
        img = cv2.imread(self.imgs[self.idx]); self.idx += 1
        padded,_,_ = letterbox(img)                          # 和推理前处理一致！
        blob = padded[:,:,::-1].transpose(2,0,1)
        blob = np.ascontiguousarray(blob,dtype=np.float32)/255.0
        cuda.memcpy_htod(self.d_in, blob)
        return [int(self.d_in)]
    def read_calibration_cache(self):
        return open(self.cache,"rb").read() if os.path.exists(self.cache) else None
    def write_calibration_cache(self, cache):
        open(self.cache,"wb").write(cache)

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()                          # TRT10：无 EXPLICIT_BATCH
parser = trt.OnnxParser(network, logger)
with open(ONNX,"rb") as f: parser.parse(f.read())
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<<30)
config.set_flag(trt.BuilderFlag.INT8)                       # 开 INT8
config.int8_calibrator = Calibrator(CALIB_DIR, CACHE)       # 挂校准器
print("INT8 校准 + 编译中（会跑一遍校准集，慢一点）...")
ser = builder.build_serialized_network(network, config)
open(ENGINE,"wb").write(ser)
print(f"完成 → {ENGINE}")
```

✅ **【检查点 D13】** `calib_images/` 放几十张 .jpg，跑通生成 `yolov8n_int8.engine` + `calib.cache`。报错 → J6。

## Day 14 · 用 INT8 引擎推理 + 初步对比

把 `infer_image.py` 的 `ENGINE` 换成 `yolov8n_int8.engine` 跑，看框准不准、记录延迟。和 FP16 对比。

## Day 15 · Debug 日：复现并修"经典 INT8 bug"

> **给 AI 老师**：这是 INT8 最著名的坑，值得专门体验。

**症状**：INT8 检测框全部堆在左上角 `(0,0)`、置信度饱和接近 1.0。
**根因**：校准器前处理和推理前处理**不一致**（比如少了 /255、或 BGR/RGB 反了、或没 letterbox）。
**修复**：确保校准器和推理**用同一套** `yolo_common` 前处理。改对后重新 `build_int8.py`。

✅ **【检查点 D15】** 学生能说出"预处理必须三处一致（训练/校准/推理）"这条规律。

🔁 **【复习 D15】** Day11 的 PTQ vs QAT 再问一遍，确保记牢。

---

# 第 4 周 · 三精度对比 + 报告 + 收尾

## Day 16-17 · 三精度基准对比（简历核心数据）

分别用 FP32/FP16/INT8 引擎跑同一批图，测**平均延迟、FPS、显存（jtop 看）、检测效果**。

```markdown
# YOLOv8n 三精度对比（Jetson Orin Nano 8GB, Super Mode）
| 精度 | 平均延迟 | FPS | 显存 | 检测效果 |
|------|---------|-----|------|---------|
| FP32 | XX ms | XX | XX MB | 基准 |
| FP16 | XX ms | XX | XX MB | 几乎无差别 |
| INT8 | XX ms | XX | XX MB | 略有漏检/轻微下降 |
结论：INT8 相比 FP32 加速约 X 倍、显存降约 X%，精度轻微下降。实际部署优先 FP16。
```

> **给 AI 老师**：严格测 mAP 要标注验证集较麻烦；教学阶段"肉眼看效果 + 延迟数字"够写简历、够讲面试。让学生**报自己实测的数字**（`jtop`/`tegrastats`），别抄博客。

## Day 18-19 · 写部署报告

`deploy_report.md`：① 环境版本（JetPack/CUDA/TRT）；② 踩坑记录（这四周的真实问题+解决）；③ 三精度对比表 + 实时 FPS。

## Day 20 · 缓冲 + Demo 视频

录一段摄像头实时检测视频（显示 FPS）。整理代码。

---

## 本阶段产出物（打勾）

- [ ] Jetson 刷机 + 摄像头点亮
- [ ] `yolo_common.py`（YOLOv8 前后处理）
- [ ] `infer_image.py` / `infer_camera.py`（手写推理）
- [ ] `build_int8.py`（手写校准器）
- [ ] 三个引擎（fp32/fp16/int8）
- [ ] `deploy_report.md`（含三精度对比）
- [ ] Demo 视频

**面试话术**：
> "我在 Jetson Orin Nano 上部署 YOLOv8，手写了 letterbox 前处理和 `[1,84,8400]` 解码+NMS 后处理（理解 v8 anchor-free、无 objectness、和 v5 输出格式的差异），FP16 实时 XX FPS。"
> "做过 FP32/FP16/INT8 对比，INT8 用 Entropy 校准器、≥1000 张校准图、batch=1，加速 X 倍、显存降 X%。踩过并修复过'预处理不一致导致框堆在(0,0)'的经典坑。PTQ 够用没上 QAT。"

---

## 报错对照表

**J1 · 刷机后 nvcc/tensorrt 找不到** — 加 PATH 到 `~/.bashrc`：`export PATH=/usr/local/cuda/bin:$PATH`、`export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`，`source ~/.bashrc`。TensorRT 用**系统 python3**测（`python3 -c "import tensorrt"`），别在 venv 里找。

**J2 · CSI 摄像头打不开** — ① `ls /dev/video*` 无输出→排线没插好/插反，断电重插看金手指。② `no element "nvarguscamerasrc"`→重启 `sudo systemctl restart nvargus-daemon`，或确认设备树里 IMX219 overlay 已启用（`sudo /opt/nvidia/jetson-io/jetson-io.py`）后重启。③ **pip 的 opencv-python 顶掉了系统 OpenCV**→摄像头脚本用系统 python3 跑，或卸载 venv 里的 opencv-python。④ 第二次打开失败（JP6.2 已知 bug）→重启 nvargus-daemon。

**J3 · `trtexec` 找不到** — 在 `/usr/src/tensorrt/bin/`。`export PATH=$PATH:/usr/src/tensorrt/bin`。

**J4 · 检测框全歪** — 后处理坐标还原错。检查 `yolo_common.postprocess`：① 先减 `dw/dh` 再除 `r`；② 有没有做 `.T` 转置（v8 必须转置）；③ preprocess 和 postprocess 用的是同一套 r/dw/dh。

**J5 · 实时 FPS 个位数** — ① 没开 Super：`sudo nvpmodel -m 0 && sudo jetson_clocks`。② 用了 FP32 引擎→换 FP16。③ 每帧重新 mem_alloc→代码里已用 `global` 只分配一次，确认没改坏。

**J6 · INT8 校准报错** — ① `calib_images/` 没 .jpg 或路径错。② 校准器前处理和推理不一致（见 Day15）。③ 显存不足→batch 保持 1。贴完整报错。

**⚠️ 版本差异** — 若 Jetson 是老 JetPack(TRT8)：`execute_async_v3`/`set_tensor_address` 换 v2 binding 写法；`create_network()` 加回 `EXPLICIT_BATCH`。先 `python3 -c "import tensorrt; print(tensorrt.__version__)"` 确认。**建议用 JetPack 6.2 免这些事。**
