# YOLO11

> 发布者: Ultralytics (2024)
>
> 特点: 下一代 YOLO 架构，C3k2 模块，全方位性能提升

---

## 一、前言

YOLO11 是 Ultralytics 于 2024 年发布的最新一代 YOLO 系列，在 YOLOv8 的基础上进行了架构革新，引入了 **C3k2 模块** 和更高效的骨干网络设计。

**核心定位：**
- 下一代 YOLO 架构
- 更高效的特征提取
- 更快的推理速度
- 更高的检测精度
- 支持检测、分割、姿态、分类、OBB 等多种任务

---

## 二、网络架构

### 2.1 整体结构

```
输入 (640×640×3)
    ↓
┌─────────────────────────────────────────┐
│  Backbone: 改进的 CSPDarknet            │
│  - C3k2 模块（核心创新）                 │
│  - 更高效的特征提取                      │
│  - 优化的通道配置                        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Neck: PANet + FPN（优化版）             │
│  - C3k2 构建的特征融合                   │
│  - 更轻量的设计                          │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Head: Decoupled Head                   │
│  - Anchor-free                           │
│  - 分类和回归分离                        │
│  - 支持多种任务输出                      │
└─────────────────────────────────────────┘
```

### 2.2 C3k2 模块（核心创新）

**C2f vs C3k2：**

```
C2f 模块 (YOLOv8)：              C3k2 模块 (YOLO11)：
                                
输入                             输入
  │                                │
  ├──→ Conv ───┬──→ Conv ──→ 输出  ├──→ Conv ───┬──→ C3k ──→ 输出
  │            │                   │            │
  │            ├──→ Conv ──→ 输出  │            └──→ C3k ──→ 输出
  │            │                   │
  │            └──→ ...            └──→ 直连 ─────────────→ 输出
  │
  └──→ Conv ───┘

C3k2 特点：
- 使用 C3k 块替代普通 Bottleneck
- C3k 可以切换为 C2f 或 C3 风格
- 更高效的梯度流动
- 可配置的扩展比例
```

**C3k 模块：**

```python
class C3k(nn.Module):
    """C3k 模块，支持两种模式"""
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, k=3):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(c1, self.c, 1, 1)
        
        # 可以配置为 C2f 风格或 C3 风格
        self.m = nn.Sequential(*[
            Bottleneck(self.c, self.c, shortcut, k=k) for _ in range(n)
        ])
        
        self.cv3 = Conv(2 * self.c, c2, 1)
    
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x1 = self.m(x1)
        return self.cv3(torch.cat([x1, x2], dim=1))
```

### 2.3 网络缩放

| 模型 | 输入尺寸 | 参数量 | mAP@50-95(COCO) | 速度 (T4) |
|:---:|:---:|:---:|:---:|:---:|
| YOLO11n | 640×640 | 2.6M | 39.5% | 1.5ms |
| YOLO11s | 640×640 | 9.4M | 47.0% | 2.5ms |
| YOLO11m | 640×640 | 20.1M | 51.5% | 4.5ms |
| YOLO11l | 640×640 | 25.3M | 53.4% | 6.2ms |
| YOLO11x | 640×640 | 56.9M | 54.7% | 11.1ms |

---

## 三、关键改进

### 3.1 C3k2 模块详解

```
C3k2 是 C2f 的改进版本：

改进点：
1. 使用 C3k 块替代普通 Bottleneck
2. C3k 支持可配置的卷积核大小
3. 更高效的通道利用
4. 更好的特征复用

配置参数：
- c1, c2: 输入输出通道
- n: C3k 块的数量
- e: 扩展比例（默认 0.5）
- k: 卷积核大小（默认 3）
- shortcut: 是否使用残差连接
```

### 3.2 优化的骨干网络

```
YOLO11 的骨干网络改进：

1. 通道配置优化
   - 更合理的通道数分配
   - 减少冗余计算
   
2. 下采样策略
   - 使用 SCDown（空间-通道解耦下采样）
   - 减少计算量
   
3. 特征提取效率
   - C3k2 块的高效堆叠
   - 更好的梯度传播
```

### 3.3 检测头优化

```
YOLO11 的检测头：

1. 解耦设计
   - 分类和回归分离
   - 各自优化
   
2. 轻量化
   - 减少头的计算量
   - 保持精度
   
3. 多任务支持
   - 检测：bbox + cls
   - 分割：+ mask
   - 姿态：+ keypoints
   - OBB：+ rotated bbox
```

---

## 四、多任务支持

### 4.1 支持的视觉任务

```
YOLO11 统一框架：

1. 目标检测 (Detection)
   - 输出: bbox + class + conf
   - 文件: yolo11n.pt

2. 实例分割 (Segmentation)
   - 输出: bbox + mask + class
   - 文件: yolo11n-seg.pt

3. 姿态估计 (Pose Estimation)
   - 输出: bbox + keypoints
   - 支持人体、动物等
   - 文件: yolo11n-pose.pt

4. 图像分类 (Classification)
   - 输出: class probabilities
   - 文件: yolo11n-cls.pt

5. 旋转目标检测 (OBB)
   - 输出: rotated bbox
   - 适合遥感、航拍
   - 文件: yolo11n-obb.pt
```

### 4.2 统一接口

```python
from ultralytics import YOLO

# 检测
model = YOLO('yolo11n.pt')
model.predict('image.jpg')

# 分割
model = YOLO('yolo11n-seg.pt')
model.predict('image.jpg')

# 姿态估计
model = YOLO('yolo11n-pose.pt')
model.predict('image.jpg')

# 分类
model = YOLO('yolo11n-cls.pt')
model.predict('image.jpg')

# OBB
model = YOLO('yolo11n-obb.pt')
model.predict('image.jpg')
```

---

## 五、性能对比

### 5.1 与 YOLOv8 对比

```
模型            参数量    mAP@50-95    速度(T4)
───────────────────────────────────────────────────
YOLOv8n         3.2M      37.3%        1.8ms
YOLO11n         2.6M      39.5%        1.5ms  ← 更快更准
───────────────────────────────────────────────────
YOLOv8s         11.2M     44.9%        3.2ms
YOLO11s         9.4M      47.0%        2.5ms  ← 更快更准
───────────────────────────────────────────────────
YOLOv8m         25.9M     50.2%        5.8ms
YOLO11m         20.1M     51.5%        4.5ms  ← 更快更准
───────────────────────────────────────────────────
YOLOv8l         43.7M     52.9%        7.9ms
YOLO11l         25.3M     53.4%        6.2ms  ← 更快更准
───────────────────────────────────────────────────
YOLOv8x         68.2M     53.9%        13.7ms
YOLO11x         56.9M     54.7%        11.1ms ← 更快更准
```

### 5.2 关键发现

```
1. 全面超越 YOLOv8：
   - 所有尺度都比 YOLOv8 快
   - 所有尺度都比 YOLOv8 准
   - 参数量更少

2. 效率提升：
   YOLO11n: 2.6M 参数 → 39.5% mAP
   YOLOv8n: 3.2M 参数 → 37.3% mAP
   参数量减少 19%，精度提升 2.2%

3. 速度提升：
   平均比 YOLOv8 快 15-20%
```

### 5.3 与其他 SOTA 对比

```
模型            参数量    mAP@50-95    速度(T4)
───────────────────────────────────────────────────
YOLOv9-T        2.0M      38.3%        -
YOLO11n         2.6M      39.5%        1.5ms
───────────────────────────────────────────────────
YOLOv9-S        7.2M      46.8%        -
YOLO11s         9.4M      47.0%        2.5ms
───────────────────────────────────────────────────
YOLOv10-S       7.2M      46.3%        2.49ms
YOLO11s         9.4M      47.0%        2.5ms
───────────────────────────────────────────────────
YOLOv10-M       15.4M     51.1%        4.74ms
YOLO11m         20.1M     51.5%        4.5ms
```

---

## 六、开发者体验

### 6.1 CLI 工具

```bash
# 安装
pip install ultralytics

# 检测
yolo predict model=yolo11n.pt source='image.jpg'

# 训练
yolo detect train data=coco.yaml model=yolo11n.pt epochs=100

# 验证
yolo detect val model=yolo11n.pt data=coco.yaml

# 导出
yolo export model=yolo11n.pt format=onnx
```

### 6.2 Python API

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolo11n.pt')

# 训练
model.train(data='coco.yaml', epochs=100, imgsz=640)

# 预测
results = model('image.jpg')
for r in results:
    print(r.boxes)      # 边界框信息
    print(r.masks)      # 分割掩码（如果支持）
    print(r.keypoints)  # 关键点（如果支持）
    r.show()            # 显示结果
    r.save()            # 保存结果

# 导出
model.export(format='onnx')
model.export(format='engine')  # TensorRT
model.export(format='tflite')  # TFLite
```

### 6.3 支持格式

```
导出格式：
- PyTorch (.pt)
- ONNX (.onnx)
- TorchScript (.torchscript)
- CoreML (.mlmodel)
- TensorRT (.engine)
- OpenVINO (.xml)
- TensorFlow (.pb, .tflite, .saved_model)
- PaddlePaddle (.pdmodel)
- NCNN (.param, .bin)
- RKNN (.rknn)
```

---

## 七、面试要点

### 7.1 高频问题

**Q1: YOLO11 相比 YOLOv8 有什么改进？**

```
答：
1. C3k2 模块
   - 替代 C2f 模块
   - 更高效的特征提取
   - 可配置的扩展比例

2. 优化的骨干网络
   - 更合理的通道配置
   - 更高效的特征提取

3. 性能提升
   - 所有尺度都比 YOLOv8 快 15-20%
   - 所有尺度都比 YOLOv8 准 1-2%
   - 参数量更少
```

**Q2: 什么是 C3k2 模块？**

```
答：
C3k2 是 YOLO11 的核心模块：

特点：
- 使用 C3k 块替代普通 Bottleneck
- C3k 可以配置为 C2f 或 C3 风格
- 支持可配置的卷积核大小
- 更高效的通道利用

优势：
- 更好的梯度流动
- 更高的特征提取效率
- 更少的参数量
```

**Q3: YOLO11 支持哪些任务？**

```
答：
YOLO11 支持 5 种视觉任务：

1. 目标检测 (Detection)
2. 实例分割 (Segmentation)
3. 姿态估计 (Pose Estimation)
4. 图像分类 (Classification)
5. 旋转目标检测 (OBB)

统一接口，切换方便
```

**Q4: YOLO11 为什么比 YOLOv8 快？**

```
答：
1. C3k2 模块更高效
   - 减少冗余计算
   - 更好的特征复用

2. 优化的通道配置
   - 更合理的通道数分配
   - 减少参数量

3. 轻量化设计
   - 检测头优化
   - 下采样优化

数据：
YOLO11n: 1.5ms
YOLOv8n: 1.8ms
快 17%
```

**Q5: YOLO11 的 C3k2 和 YOLOv8 的 C2f 有什么区别？**

```
答：
C2f (YOLOv8)：
- 使用普通 Bottleneck
- 多个梯度流分支

C3k2 (YOLO11)：
- 使用 C3k 块
- C3k 可以切换风格（C2f/C3）
- 可配置卷积核大小
- 更高效的特征提取

改进：
- 参数量减少 19%
- 精度提升 2.2%
```

### 7.2 手撕代码要点

```python
# C3k 模块
class C3k(nn.Module):
    """C3k 模块"""
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, k=3):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(c1, self.c, 1, 1)
        
        # C3k 块序列
        self.m = nn.Sequential(*[
            Bottleneck(self.c, self.c, shortcut, k=k) 
            for _ in range(n)
        ])
        
        self.cv3 = Conv(2 * self.c, c2, 1)
    
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x1 = self.m(x1)
        return self.cv3(torch.cat([x1, x2], dim=1))


# C3k2 模块
class C3k2(nn.Module):
    """C3k2 模块 - YOLO11 核心"""
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, k=3):
        super().__init__()
        self.c = int(c2 * e)
        
        # 输入分割
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        
        # C3k 块序列
        self.m = nn.ModuleList([
            C3k(self.c, self.c, n=1, shortcut=True, e=1.0, k=k) 
            for _ in range(n)
        ])
        
        # 输出融合
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# SCDown（空间-通道解耦下采样）
class SCDown(nn.Module):
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        # 逐通道深度卷积下采样
        self.dwconv = nn.Conv2d(c1, c1, k, s, k//2, groups=c1)
        # 点卷积调整通道
        self.pwconv = nn.Conv2d(c1, c2, 1)
    
    def forward(self, x):
        return self.pwconv(self.dwconv(x))
```

---

## 八、总结

| 特性 | YOLO11 特点 |
|:---:|:---|
| **C3k2 模块** | 核心创新，更高效特征提取 |
| **优化骨干** | 更合理的通道配置 |
| **全面超越** | 比 YOLOv8 快 15-20%，精度更高 |
| **多任务** | 检测、分割、姿态、分类、OBB |
| **开发体验** | 简洁的 API 和 CLI |
| **导出支持** | 12+ 种部署格式 |

**一句话：** YOLO11 通过引入 C3k2 模块和全面优化的架构，在更少的参数量下实现了比 YOLOv8 更快的速度和更高的精度，是 YOLO 系列的最新 SOTA。

---

## 参考资源

- 官方文档: https://docs.ultralytics.com/models/yolo11/
- GitHub: https://github.com/ultralytics/ultralytics
- 快速开始: https://docs.ultralytics.com/quickstart/
