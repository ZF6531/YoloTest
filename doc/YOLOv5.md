# YOLOv5

> 发布者: Ultralytics (2020)
>
> 特点: PyTorch 原生实现，工程优化极佳

---

## 一、前言

**重要说明：**

YOLOv5 是由 **Ultralytics** 发布的版本，**不是由 YOLO 原作者 (Joseph Redmon/Alexey Bochkovskiy) 发布**。它是基于 YOLOv3 和 YOLOv4 的思想，用 PyTorch 重新实现的版本。

尽管如此，YOLOv5 因其**优秀的工程实现、易用性和性能**，在工业界获得了广泛应用。

---

## 二、网络架构

### 2.1 整体结构

YOLOv5 的网络结构与 YOLOv4 类似，但使用 PyTorch 原生实现，并做了大量工程优化。

```
输入图像 (640×640×3)
    ↓
┌─────────────────────────────────────────┐
│  Backbone: CSPDarknet (改进版)          │
│  - Focus 切片结构                        │
│  - CSP 块                                │
│  - SPPF (快速空间金字塔池化)             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Neck: PANet                            │
│  - FPN + PAN 结构                        │
│  - CSP 化                                │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Head: YOLOv5 Head                       │
│  - 三个尺度预测                          │
│  - Anchor-based                          │
│  - 解耦检测头 (v6.0+)                    │
└─────────────────────────────────────────┘
```

### 2.2 Focus 切片结构（创新点）

**核心思想：** 用切片操作代替下采样卷积，减少计算量

```
传统下采样：
输入 640×640×3 ──→ Conv(stride=2) ──→ 320×320×32
    计算量: 640×640×3×32×3×3 = 353,894,400

Focus 切片：
输入 640×640×3
    │
    ├──→ 每隔一个像素取一个值，分成4份
    │
    ├─→ 左上角: 320×320×3
    ├─→ 右上角: 320×320×3
    ├─→ 左下角: 320×320×3
    └─→ 右下角: 320×320×3
    │
    └──→ 通道拼接: 320×320×12
    │
    └──→ Conv(1×1, 32) ──→ 320×320×32
    计算量: 320×320×12×32×1×1 = 39,321,600
    
计算量减少约 89%！
```

**代码实现：**

```python
class Focus(nn.Module):
    """Focus wh information into c-space"""
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s)
    
    def forward(self, x):
        # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([
            x[..., ::2, ::2],    # 左上角
            x[..., 1::2, ::2],   # 右上角
            x[..., ::2, 1::2],   # 左下角
            x[..., 1::2, 1::2]   # 右下角
        ], dim=1))
```

### 2.3 CSP 结构

```
YOLOv5 中的 CSP 块（C3 模块）：

输入
  │
  ├──→ Conv 1×1 ──┐
  │               ├──→ Concat ──→ Conv 1×1 ──→ 输出
  └──→ Conv 1×1 ──┤   ↑
            ↓     │   │
      ┌───────────┘   │
      │ ResBlock ×N   │
      │ (Bottleneck)  │
      └───────────────┘

v6.0 后使用 C3 模块替代原 CSP：
- 去掉残差连接中的一个 1×1 Conv
- 更轻量，速度更快
```

### 2.4 SPPF (Spatial Pyramid Pooling Fast)

```
传统 SPP：                       SPPF：
                                  
输入                             输入
  │                                │
  ├──→ MaxPool(5×5) ──┐            │
  ├──→ MaxPool(9×9) ──┼──→ Concat  ├──→ MaxPool(5×5) ──→ MaxPool(5×5) ──→ MaxPool(5×5)
  ├──→ MaxPool(13×13)─┤            │         │                │                │
  └──→ 原图 ──────────┘            │         └────────────────┴────────────────┘
                                   │                         │
                                   └─────────────────────────┴──→ Concat ──→ 输出

SPPF 用串行小池化代替并行大池化，结果相同但速度更快
5×5 + 5×5 = 9×9 感受野
5×5 + 5×5 + 5×5 = 13×13 感受野
```

### 2.5 网络缩放（Scale）

YOLOv5 提供了不同大小的模型：

| 模型 | 宽度因子 | 深度因子 | 参数量 | mAP@0.5 | FPS (T4) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| YOLOv5n (nano) | 0.25 | 0.33 | 1.9M | 45.7% | 1537 |
| YOLOv5s (small) | 0.50 | 0.33 | 7.2M | 56.8% | 636 |
| YOLOv5m (medium) | 0.75 | 0.67 | 21.2M | 64.1% | 417 |
| YOLOv5l (large) | 1.00 | 1.00 | 46.5M | 67.3% | 294 |
| YOLOv5x (xlarge) | 1.25 | 1.33 | 86.7M | 68.9% | 196 |

```
缩放方式：
- 宽度因子：控制通道数 (c = c_base × width_factor)
- 深度因子：控制层数 (n = n_base × depth_factor)
```

---

## 三、训练技巧与优化

### 3.1 AutoAnchor

```
问题：不同数据集的物体尺寸分布不同，默认 anchor 可能不合适

解决：AutoAnchor
1. 分析训练集中所有 ground truth 框的分布
2. 使用遗传算法优化 anchor 尺寸
3. 确保 anchor 与数据集匹配

进化目标：
- anchor 与 ground truth 的最佳可能召回率 (BPR) > 0.98
- 如果达不到，自动调整 anchor
```

### 3.2 马赛克增强 + MixUp

```python
# YOLOv5 的数据增强流程
1. Mosaic (概率 1.0)：4 图拼接
2. MixUp (概率 0.1)：两张图融合
3. 随机仿射变换：旋转、平移、缩放、剪切
4. HSV 增强：色调、饱和度、亮度
5. 随机水平翻转
6. 随机擦除 (Cutout)
```

### 3.3 损失函数

```python
# YOLOv5 损失 = 3 部分
L_total = λ_box × L_box + λ_obj × L_obj + λ_cls × L_cls

L_box: CIoU Loss (边界框回归)
L_obj: BCE Loss (目标置信度)
L_cls: BCE Loss (分类)

权重：
- YOLOv5n/s: (0.05, 1.0, 0.5)
- YOLOv5m/l/x: (0.05, 1.0, 0.5)
```

### 3.4 训练策略

```
1. Warmup：前 3 个 epoch 线性增加学习率
2. Cosine Annealing：余弦退火学习率衰减
3. EMA (Exponential Moving Average)：模型参数滑动平均
4. Mixed Precision：混合精度训练 (FP16)
5. Multi-scale：每 10 batch 随机改变输入尺寸 (±50%)
```

---

## 四、版本演进

### 4.1 YOLOv5 版本历史

| 版本 | 时间 | 主要更新 |
|:---:|:---:|:---|
| v1.0 | 2020.06 | 初始发布 |
| v3.0 | 2020.11 | 改进损失函数，添加 P6 模型 |
| v4.0 | 2021.04 | SPPF，更好的数据增强 |
| v5.0 | 2021.06 | 改进模型结构 |
| v6.0 | 2021.11 | **重大更新**：解耦检测头，Anchor-free 选项 |
| v6.1 | 2022.02 | TensorRT 优化 |
| v7.0 | 2022.11 | 分割模型支持 |
| v8.0 | 2023.01 | **YOLOv8 发布**（新一代） |

### 4.2 v6.0 重大更新

**解耦检测头 (Decoupled Head)：**

```
传统耦合头：                       解耦头 (v6.0+)：
                                  
特征图                             特征图
  │                                  │
  ├──→ Conv ──→ (cls+box+obj)        ├──→ 分类分支：Conv ──→ Conv ──→ cls
                                    │
                                    └──→ 回归分支：Conv ──→ Conv ──→ box
                                    
耦合头：一个分支同时预测分类和回归
解耦头：两个独立分支分别预测

优点：
- 收敛更快
- 精度更高 (+1-2% mAP)
- 小模型提升更明显
```

---

## 五、性能对比

```
方法           输入尺寸    参数量    mAP@0.5(COCO)   FPS (V100)
─────────────────────────────────────────────────────────────────
YOLOv5n        640×640     1.9M      45.7%          1537
YOLOv5s        640×640     7.2M      56.8%          636
YOLOv5m        640×640     21.2M     64.1%          417
YOLOv5l        640×640     46.5M     67.3%          294
YOLOv5x        640×640     86.7M     68.9%          196
─────────────────────────────────────────────────────────────────
YOLOv4         608×608     64.4M     65.7%          62
EfficientDet-D0 512×512    3.9M      52.6%          62
─────────────────────────────────────────────────────────────────
```

**结论：**
- YOLOv5s：小模型中的速度精度王者
- YOLOv5m/l：兼顾速度和精度
- YOLOv5x：精度最高，适合离线场景

---

## 六、工程优势

### 6.1 为什么 YOLOv5 受欢迎？

```
1. PyTorch 原生实现
   - 易于理解和修改
   - 与生态兼容性好

2. 完善的工程支持
   - 自动下载 COCO 等数据集
   - 自动多卡训练
   - 多种导出格式 (ONNX, TensorRT, CoreML, etc.)

3. 丰富的预训练模型
   - 5 种尺度
   - P5/P6 版本
   - 分类/检测/分割

4. 优秀的文档和社区
   - 详尽的教程
   - 活跃的 issue 讨论

5. 易用的命令行接口
   train.py / detect.py / export.py
```

### 6.2 导出与部署

```bash
# 训练
python train.py --data coco.yaml --weights yolov5s.pt

# 检测
python detect.py --source image.jpg --weights yolov5s.pt

# 导出
python export.py --weights yolov5s.pt --include onnx torchscript

# 支持格式：
# - PyTorch (.pt)
# - ONNX (.onnx)
# - TorchScript (.torchscript)
# - TensorRT (.engine)
# - CoreML (.mlmodel)
# - OpenVINO (.xml)
# - TensorFlow (.pb, .tflite, .saved_model)
```

---

## 七、面试要点

### 7.1 高频问题

**Q1: YOLOv5 是谁发布的？和原版 YOLO 什么关系？**

```
答：
- YOLOv5 由 Ultralytics 公司发布
- 不是 YOLO 原作者 (Joseph Redmon) 发布的
- 是基于 YOLOv3/v4 的思想，用 PyTorch 重新实现的
- 因工程优化好、易用性强，在工业界广泛使用
```

**Q2: YOLOv5 的 Focus 结构是什么？有什么优势？**

```
答：
Focus 是一种切片下采样操作：
1. 将 640×640×3 的图像每隔一个像素切片
2. 得到 4 个 320×320×3 的特征图
3. 通道拼接成 320×320×12
4. 经过 1×1 卷积得到 320×320×32

优势：
- 计算量比 stride=2 卷积减少约 89%
- 信息不丢失（每个像素都用到了）
- 速度更快
```

**Q3: YOLOv5 和 YOLOv4 的主要区别？**

```
答：
网络结构：
- 类似，都使用 CSPDarknet + PANet
- YOLOv5 使用 Focus 替代下采样
- YOLOv5 使用 SPPF 替代 SPP（更快）

实现：
- YOLOv5: PyTorch
- YOLOv4: Darknet

工程：
- YOLOv5 有更好的工程实现（易用性、导出、部署）
- 提供多种尺度模型 (n/s/m/l/x)
- 提供 P5/P6 版本
```

**Q4: YOLOv5 的 AutoAnchor 是什么？**

```
答：
- 自动分析训练集，优化 anchor 尺寸
- 使用遗传算法寻找最优 anchor
- 目标是最佳可能召回率 (BPR) > 0.98
- 使 anchor 更好地匹配数据集的目标分布
```

**Q5: YOLOv5 v6.0 的解耦检测头是什么？**

```
答：
传统耦合头：一个分支同时输出分类和回归
解耦头：两个独立分支分别输出

优点：
- 收敛更快
- 精度更高（小模型提升更明显）
- 训练和推理更稳定
```

### 7.2 手撕代码要点

```python
# Focus 结构
class Focus(nn.Module):
    def forward(self, x):
        # 切片操作
        return self.conv(torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], 1))

# SPPF（快速空间金字塔）
class SPPF(nn.Module):
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)      # 5×5 MaxPool
        y2 = self.m(y1)     # 又一个 5×5 (等效 9×9)
        y3 = self.m(y2)     # 再一个 5×5 (等效 13×13)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

# C3 模块
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c1, c2//2, 1, 1)
        self.cv2 = Conv(c1, c2//2, 1, 1)
        self.cv3 = Conv(c2, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c2//2, c2//2, shortcut) for _ in range(n)])
    
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
```

---

## 八、总结

| 特性 | YOLOv5 优势 |
|:---:|:---|
| **工程实现** | PyTorch 原生，代码清晰易读 |
| **Focus 结构** | 创新的切片下采样，减少计算量 |
| **模型缩放** | n/s/m/l/x 五种尺度，灵活选择 |
| **易用性** | 完善的 CLI、自动下载、多格式导出 |
| **社区** | 活跃的社区，丰富的教程和预训练模型 |

**一句话：** YOLOv5 是 YOLO 系列中最工程化的版本，通过 Focus 结构、完善的工具和丰富的模型选择，成为工业界目标检测的首选方案之一。

---

## 参考资源

- 官方仓库: https://github.com/ultralytics/yolov5
- 文档: https://docs.ultralytics.com/yolov5/
- 论文解读: https://zhuanlan.zhihu.com/p/172125428
