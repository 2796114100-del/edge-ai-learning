# 04 · 阶段四：ROS2 节点 + 串口 + Arduino 追踪（第 12-15 周）

> **给 AI 老师**：先读 `00-教学总纲.md`。本阶段把推理接到真实硬件，做**相机→推理→串口→舵机追踪**的端到端闭环。**下位机用 Arduino 不用 STM32**（学生目标是推理部署岗，不是嵌入式岗；Arduino 半小时搞定、效果一样）。**把 AI 半和执行半分开做，最后再合体**——先各自跑通，联调才不抓瞎。

**基准环境**：Jetson Orin Nano + **JetPack 6.2**（Ubuntu 22.04 → **ROS2 Humble 可原生 apt 安装**）+ YOLOv8 engine（阶段二产物）+ Arduino + 2 舵机云台。

> ⚠️ **给 AI 老师的三条铁提醒**：
> 1. **JetPack 6 = Ubuntu 22.04 → Humble 原生装**（`apt install ros-humble-*`）。JetPack 5 是 20.04 装不了 Humble。
> 2. **Jetson 上最大坑是 OpenCV 版本冲突**（系统 OpenCV vs ROS 的 vs pip 的），表现为链接错误或**运行时段错误**。规矩：**全项目统一一个 OpenCV，用 `ldd` 验证**。见 J3。
> 3. **Arduino Uno/Nano 走 USB 是 `/dev/ttyACM0`**（不是 ttyUSB0！ttyUSB0 是 CH340/FTDI 芯片的）。且**开串口会触发 Arduino 自动复位**（丢约 1.5s 数据）——开口后等 ~2s 再发。见 J4。

---

## 本阶段目标

```
装 ROS2 Humble → 写 C++ 订阅节点 → 自定义消息 TargetError → 检测节点(订阅图,发目标偏差)
→ Arduino 舵机 + 串口接收 → C++ 控制节点(P控制→串口发角度) → 合体实时追踪
```

**学完能做到**：写 ROS2 C++ 节点；把推理集成进机器人系统；做出"看得见"的端到端追踪 Demo（简历杀手锏）。

**时间**：约 4 周。

---

## 参考资源（引用务必带【适配提示】）

| 资源 | 地址 | 用途 |
|------|------|------|
| ROS2 Humble 官方 C++ pub/sub | https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html | 核心必做 |
| 古月居 ROS2入门21讲 | https://github.com/guyuehome/ros2_21_tutorials | B站搜"古月居 ROS2入门21讲"，结构最好 |
| 鱼香ROS 动手学ROS2 | https://fishros.com/d2lros2/ | 极友好，一键装 `wget http://fishros.com/install` |
| mgonzs13/yolo_ros | https://github.com/mgonzs13/yolo_ros | **最活跃**的 ROS2 YOLO 封装，支持 TensorRT engine，抄它的消息/launch 设计 |
| cyrusbehr/YOLOv8-TensorRT-CPP | https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP | 干净的 C++ TRT YOLOv8 推理核，直接搬进节点 |
| WyattAutomation 云台追踪 | https://github.com/WyattAutomation/YOLOv3-ROS-Robotic-Headshot-Turret | 最完整的"相机→YOLO→ROS→云台"端到端参照（抄架构+对中逻辑） |
| Robin2 Serial Input Basics | https://forum.arduino.cc/t/serial-input-basics-updated/382007 | **串口防粘包**的社区标准写法 |

> 🔧 **【适配提示 · 通用】** ① `yolo_ros` 是 **Python** 封装、YOLOv8——你要 C++ 的话抄它的**消息设计和 launch 结构**，推理核用 cyrusbehr 的 C++ TRT。② 云台追踪参照多是 **YOLOv3/USB 摄像头/非 Jetson**——抄**控制逻辑（对中→P控制→串口）**，检测换成你的 YOLOv8+TRT、摄像头用阶段二的 CSI 管道。③ 古月居/鱼香有 ROS1 也有 ROS2，**认准 ROS2/Humble 的**。

---

# 第 1 周 · ROS2 基础（先不碰 AI）

## Day 1 · 装 ROS2 Humble + 验证

```bash
# JetPack 6（Ubuntu 22.04）原生装
sudo apt update && sudo apt install -y ros-humble-ros-base ros-humble-cv-bridge ros-humble-vision-opencv
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && source ~/.bashrc
# 验证：两个终端
ros2 run demo_nodes_cpp talker      # 终端1
ros2 run demo_nodes_cpp listener    # 终端2
```

✅ **【检查点 D1 · 版本确认】** talker 发、listener 收到 `Hello World`。报错 → R1。

> **【原理速讲】Topic = 广播电台**：一个节点往频道（如 `/camera/image`）发，订阅该频道的节点都能收，彼此不用认识、只认频道名。这是机器人系统"解耦"的核心。

## Day 2 · 概念 + CLI 工具

跟古月居"ROS2入门21讲"看 nodes/topics 两章。练 `ros2 node list`、`ros2 topic list/echo`、`rqt_graph`。

## Day 3 · 建工作空间 + 第一个 C++ 订阅节点

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake inference_node --dependencies rclcpp sensor_msgs cv_bridge
```

`~/ros2_ws/src/inference_node/src/main.cpp`：

```cpp
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
class InferenceNode : public rclcpp::Node {
public:
    InferenceNode() : Node("inference_node") {
        sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/image", 10,
            [this](const sensor_msgs::msg::Image::SharedPtr msg){
                RCLCPP_INFO(get_logger(), "收到图像 %dx%d", msg->width, msg->height);
            });
        RCLCPP_INFO(get_logger(), "推理节点已启动");
    }
private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
};
int main(int argc, char** argv){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<InferenceNode>());
    rclcpp::shutdown(); return 0;
}
```

`CMakeLists.txt` 加（`ament_package()` 前）：
```cmake
add_executable(inference_node src/main.cpp)
ament_target_dependencies(inference_node rclcpp sensor_msgs cv_bridge)
install(TARGETS inference_node DESTINATION lib/${PROJECT_NAME})
```

```bash
cd ~/ros2_ws && colcon build --packages-select inference_node
source install/setup.bash && ros2 run inference_node inference_node
```

✅ **【检查点 D3】** 打印"推理节点已启动"。`colcon build` 失败 → R2。

## Day 4 · 自己写 pub/sub + 参数/launch

跟鱼香"动手学ROS2"练：写一个发布者发数字、订阅者收；学 launch 文件。

## Day 5 · 自定义消息 TargetError（检测↔控制的契约）

建 `.msg`：`TargetError.msg`：
```
float32 x_err     # 目标中心相对画面中心的水平偏差(归一化 -1~1)
float32 y_err     # 垂直偏差
bool found        # 是否检测到目标
```
配好 `rosidl` 依赖，`colcon build`，`ros2 interface show` 能看到。

✅ **【检查点 D5】** 能 build 并 echo 自定义消息。🔁 复习：Topic 通信是谁发谁收？

---

# 第 2 周 · 视觉进 ROS2 + TensorRT 推理

## Day 6 · 相机进 ROS2

用 `usb_cam` 或写个 CSI 发布节点（复用阶段二 GStreamer 管道）把帧发到 `/camera/image`；`ros2 run rqt_image_view` 看画面。

## Day 7 · cv_bridge：ROS 图 → cv::Mat

```cpp
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
// 回调里：只读用 toCvShare（零拷贝，喂推理）
void cb(const sensor_msgs::msg::Image::ConstSharedPtr& msg){
    cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;   // 送去推理
    // 要在图上画框改动时，用 toCvCopy（深拷贝）
}
```

✅ **【检查点 D7】** 订阅图、转成 cv::Mat、`imshow` 出来。**在这里把 OpenCV 冲突彻底解决**（`ldd` 验证只有一个 OpenCV）→ R3。

> **【原理速讲】** `toCvShare` 零拷贝（只读、快），`toCvCopy` 深拷贝（要修改图时用）。编码用 `"bgr8"`。

## Day 8 · YOLOv8 engine 就绪

确认阶段二的 `yolov8n.engine` 在 Jetson 上能用（或重新 `yolo export ... format=engine`）。单测 FPS。

## Day 9 · 快速起一个检测（抄仓库）

先用 `mgonzs13/yolo_ros`（Python）快速看到 ROS2 里发检测框，理解消息流。再决定 C++ 化。

## Day 10 · C++ 检测节点：订阅图→TRT推理→发 TargetError

把阶段三的 C++ TRT 推理 + 阶段二的前后处理封装成 `TrtYolo` 类，在回调里：
```cpp
cv::Mat frame = cv_bridge::toCvShare(msg,"bgr8")->image;
auto dets = trt_->infer(frame);                    // 复用阶段二/三逻辑
// 取最高分目标，算它中心相对画面中心的偏差，发 TargetError
if(!dets.empty()){
    auto& d = dets[0];
    float cx = (d.x1+d.x2)/2.0f, cy=(d.y1+d.y2)/2.0f;
    msg_out.x_err = (cx - frame.cols/2.0f)/(frame.cols/2.0f);
    msg_out.y_err = (cy - frame.rows/2.0f)/(frame.rows/2.0f);
    msg_out.found = true;
}
pub_->publish(msg_out);
```
CMakeLists 补 `find_package(CUDA)` + link `nvinfer nvinfer_plugin cudart`。

✅ **【检查点 D10 · 里程碑】** 检测节点发出 `TargetError`（`ros2 topic echo`看得到偏差随物体移动变化）。冲突 → R3/R6。

---

# 第 3 周 · 串口 + Arduino（先把执行半单独做通）

## Day 11 · Arduino 舵机（硬编码角度先动起来）

```cpp
#include <Servo.h>
Servo pan, tilt;
void setup(){ pan.attach(9); tilt.attach(10); }
void loop(){ pan.write(90); tilt.write(90); delay(1000); pan.write(45); delay(1000); }
```
✅ 舵机能转、机械行程 OK、`constrain` 限位。

## Day 12 · Arduino 串口接收（Robin2 防粘包写法）

```cpp
// arduino_servo.ino —— 收 "<pan,tilt>" 控制两舵机（Robin2 起止标记，非阻塞、稳）
#include <Servo.h>
Servo pan, tilt;
const byte NUM=32; char buf[NUM]; bool newData=false;
void recv(){
    static bool inProg=false; static byte i=0;
    while(Serial.available()>0 && !newData){
        char c=Serial.read();
        if(inProg){
            if(c!='>'){ buf[i]=c; if(++i>=NUM) i=NUM-1; }
            else { buf[i]='\0'; inProg=false; i=0; newData=true; }
        } else if(c=='<') inProg=true;
    }
}
void setup(){ Serial.begin(115200); pan.attach(9); tilt.attach(10); }
void loop(){
    recv();
    if(newData){
        int p=atoi(strtok(buf,",")), t=atoi(strtok(NULL,","));
        pan.write(constrain(p,0,180)); tilt.write(constrain(t,0,180));   // 限位保护
        newData=false;
    }
}
```
从 Arduino 串口监视器发 `<90,45>` 测试。

✅ **【检查点 D12】** 发 `<90,45>` 舵机到位。

> **【原理速讲】为什么用 `<...>` 起止标记？** 串口是字节流会粘连，靠 `<` 开头 `>` 结尾切分每帧，比 `readStringUntil` 更稳、非阻塞。

## Day 13 · Jetson 端 C++ 串口（termios，独立测试）

```cpp
// serial_sender.hpp
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstdio>
#include <string>
class SerialSender {
public:
    SerialSender(const char* dev="/dev/ttyACM0"){    // Arduino Uno/Nano=ACM0; CH340=USB0
        fd_ = open(dev, O_RDWR | O_NOCTTY);
        if(fd_<0){ perror("open serial"); return; }
        struct termios t{}; tcgetattr(fd_,&t);        // 必须先读现有设置
        cfsetospeed(&t,B115200); cfsetispeed(&t,B115200);
        t.c_cflag &= ~PARENB; t.c_cflag &= ~CSTOPB;   // 8N1
        t.c_cflag &= ~CSIZE;  t.c_cflag |= CS8;
        t.c_cflag &= ~CRTSCTS; t.c_cflag |= (CREAD|CLOCAL);
        t.c_lflag &= ~(ICANON|ECHO|ECHOE|ISIG);       // 原始模式
        t.c_iflag &= ~(IXON|IXOFF|IXANY|INLCR|ICRNL);
        t.c_oflag &= ~OPOST;
        t.c_cc[VMIN]=0; t.c_cc[VTIME]=1;
        tcsetattr(fd_,TCSANOW,&t);
        usleep(2000000);                               // 等 Arduino 自动复位(~1.5s)完成
    }
    void send(int pan,int tilt){
        char b[32]; int n=snprintf(b,sizeof(b),"<%d,%d>\n",pan,tilt);
        write(fd_,b,n);
    }
    ~SerialSender(){ if(fd_>=0) close(fd_); }
private: int fd_=-1;
};
```
独立 main 里 sweep 角度测试。

✅ **【检查点 D13】** Jetson C++ 程序能让舵机扫动。`Permission denied` → R4；不动/乱抖 → R5。

## Day 14 · C++ 控制节点（P 控制）

订阅 `TargetError` → P 控制算增量 → 限幅 → 串口发绝对角度（固定 20-30Hz）：
```cpp
// 收到 TargetError：
if(msg->found){
    pan_angle_  -= Kp_ * msg->x_err * 90;    // 偏右→左转(符号按你云台方向调)
    tilt_angle_ += Kp_ * msg->y_err * 90;
    pan_angle_  = std::clamp(pan_angle_, 0.f, 180.f);
    tilt_angle_ = std::clamp(tilt_angle_, 0.f, 180.f);
    serial_->send((int)pan_angle_, (int)tilt_angle_);
}
```

## Day 15 · 闭环台架测试（不接相机，手动发假偏差）

`ros2 topic pub` 手动发 `TargetError`，看舵机响应；调发送频率、确认不粘包。

> **【原理速讲】P 控制**：偏差越大转越多（`增量 = Kp × 偏差`）。Kp 太大会来回震荡，太小反应慢——Day17 现场调。发**绝对角度**（非累加）这样丢一帧不会累积跑偏。

---

# 第 4 周 · 合体 + 调优 + Demo

## Day 16 · 全链路接通

一个 launch 启动：CSI 相机节点 → 检测节点 → 控制节点 → 串口 → Arduino → 舵机。

## Day 17 · 首次实时追踪 + 调 Kp

在相机前移动物体，舵机跟随。**大概率震荡**——调小 Kp、加中心死区（偏差很小就不动）、限每次最大步进。

✅ **【检查点 D17 · 最终里程碑】** 物体移动、舵机平滑跟随。**录视频！** 简历/面试最有冲击力的东西。

## Day 18 · 平滑 + 丢失处理

加低通/限速让运动更顺；"目标丢失"时保持或缓慢回中；多目标时选最高分/最近。

## Day 19 · 鲁棒性

串口断开重连、限舵机速率、打日志记 FPS/延迟；`jtop` 确认持续实时。

## Day 20 · 打磨 + README + Demo

写 README（启动命令 + 接线图），录完整 Demo 视频。

> **给 AI 老师**：Day7(OpenCV 冲突) 和 Day17(调 Kp) 最可能超时，预留缓冲。

---

## 本阶段产出物（打勾）

- [ ] `~/ros2_ws/`（colcon build 通过）
- [ ] `TargetError.msg` + 检测节点 + 控制节点（C++）
- [ ] `serial_sender.hpp` + `arduino_servo.ino`
- [ ] **端到端追踪 Demo 视频**

**面试话术**：
> "我写了 ROS2 C++ 节点：订阅相机话题、调 TensorRT 推理、发布自定义 TargetError 消息，控制节点用 P 控制把偏差转成舵机角度、经串口(起止帧协议)发给 Arduino，做出相机→推理→执行的实时追踪闭环。踩过并解决了 Jetson 上 OpenCV 多版本冲突和 Arduino 串口自动复位的坑。"

---

## 报错对照表

**R1 · talker/listener 通不上** — 每个新终端要 `source /opt/ros/humble/setup.bash`（已写进 .bashrc 就重开终端）。

**R2 · `colcon build` 失败** — ① 要在 `~/ros2_ws` 根目录跑。② `package.xml` 依赖没声明全。③ CMakeLists 的 `ament_target_dependencies` 少包。贴报错最后 20 行。

**R3 · cv_bridge 链接失败 / 运行时段错误（Jetson 头号坑）** — 多个 OpenCV 版本混用。规矩：**统一一个 OpenCV**。CMake 里 `find_package(OpenCV REQUIRED)` 前 `set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)`；用 `ldd 你的节点 | grep opencv` 和 `ldd cv_bridge的.so` 对比，**版本/路径必须一致**。仍冲突就从源码重编 cv_bridge（clone `ros-perception/vision_opencv` humble 分支进工作空间 colcon build）。

**R4 · 串口 `Permission denied`** — `sudo usermod -aG dialout $USER` 后**注销重登**。临时：`sudo chmod 666 /dev/ttyACM0`。

**R5 · 舵机不动/乱抖** — ① 舵机用 Arduino 供电了→换独立 5V、共地。② 波特率两边都要 115200。③ 帧格式对不上→Arduino 串口监视器打印收到的原始串。④ **忘了开口后等 Arduino 复位**（代码里已 `usleep(2s)`）。

**R6 · ROS2 + TensorRT + OpenCV 一起编译冲突** — 推理逻辑先在独立程序验证好再搬进节点；CMake 里 CUDA/TRT 的 include/link 显式写全；TensorRT 本身不需要 OpenCV，前处理和节点用**同一个** OpenCV。

**设备名提醒** — Arduino Uno/Nano(USB)=`/dev/ttyACM0`；CH340/FTDI=`/dev/ttyUSB0`；Orin 40 针 UART=`/dev/ttyTHS1`。`ls /dev/ttyACM* /dev/ttyUSB*` 确认。
