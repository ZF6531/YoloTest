# YOLOv4

> 论文: "YOLOv4: Optimal Speed and Accuracy of Object Detection" (arXiv 2020)
>
> 作者: Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao

---

## 一、网络架构

### 1.1 整体结构

YOLOv4 是**第一个非原作者发布的 YOLO 版本**，目标是找到检测器的"最优组合"，引入了 BoF (Bag of Freebies) 和 BoS (Bag of Specials) 的概念。

```
输入图像 (608×608×3)
    ↓
┌─────────────────────────────────────────┐
│  Backbone: CSPDarknet53                 │
│  - Cross Stage Partial Network          │
│  - 减少计算量同时保持精度                │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Neck: SPP + PANet                      │
│  - SPP: 空间金字塔池化，增大感受野       │
│  - PANet: 双向特征金字塔，增强融合       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Head: YOLOv3 Head                      │
│  - 三个尺度预测                         │
│  - Anchor-based                         │
└─────────────────────────────────────────┘
```

### 1.2 组件总览

| 组件 | YOLOv4 选择 | 说明 |
|:---:|:---:|:---|
| **Backbone** | CSPDarknet53 | 跨阶段局部网络，减少重复梯度 |
| **Neck** | SPP + PANet | 空间金字塔 + 双向特征融合 |
| **Head** | YOLOv3 Head | 保持不变，三个尺度预测 |
| **BoF (Backbone)** | CutMix, Mosaic, DropBlock, Label Smoothing | 训练技巧 |
| **BoS (Backbone)** | Mish, CSP, MiWRC | 特殊模块 |
| **BoF (Detector)** | CIoU Loss, CmBN, DropBlock, Mosaic | 检测器训练技巧 |
| **BoS (Detector)** | Mish, SPP, SAM, PANet, DIoU-NMS | 检测器特殊模块 |

### 1.3 CSPDarknet53 骨干网络

**CSP (Cross Stage Partial) 结构：**

```
传统 ResNet 块:                 CSP 块:
┌─────────────┐                 ┌─────────────────────────────────┐
│  输入 x      │                 │  输入 x ───┬──────────────────┐ │
│    │        │                 │            │ Split            │ │
│    ↓        │                 │            ▼                  │ │
│ Conv 1×1    │                 │      ┌──────────┐             │ │
│    ↓        │                 │      │  Part 1  │ → 直接传递   │ │
│ Conv 3×3    │                 │      │  (50%)   │   到 Concat  │ │
│    ↓        │                 │      └────┬─────┘             │ │
│ 残差连接     │                 │           │ Part 2            │ │
│    ↓        │                 │           ▼ (50%)             │ │
│  输出        │                 │      ┌──────────┐             │ │
└─────────────┘                 │      │  ResNet  │             │ │
                                │      │  Blocks  │             │ │
                                │      └────┬─────┘             │ │
                                │           └──────→ Concat ────┤ │
                                │                         ↓     │ │
                                │                      Transition│
                                │                         ↓     │ │
                                │                       输出     │ │
                                └─────────────────────────────────┘

优势：
1. 减少重复梯度信息
2. 减少计算量（约20%）
3. 保持精度甚至提升
```

### 1.4 Neck: SPP + PANet

**SPP (Spatial Pyramid Pooling)：**

```
输入特征图
    │
    ├──→ MaxPool(13×13) ──┐
    ├──→ MaxPool(9×9) ────┼──→ Concat ──→ 输出
    ├──→ MaxPool(5×5) ────┤
    └──→ 直接传递 ─────────┘

作用：
- 增大感受野（不同尺度池化）
- 提取多尺度上下文
- 分离重要上下文特征（不降低速度）
```

**PANet (Path Aggregation Network)：**

```
传统 FPN（自顶向下）：             PANet（双向）：
                                  
P5 ─────────────────→ Out5        P5 ────────────────→ Out5
↑                                  ↑    ↓
P4 ─────────→ Out4                 P4 ←────────────→ Out4
↑                                  ↑    ↓
P3 ──→ Out3                        P3 ────────────────→ Out3

PANet 增加自底向上路径，增强定位信息传递
```

### 1.5 完整网络结构

```
输入 608×608×3
│
├─→ CSPDarknet53
│   ├─→ Conv(3×3, 32)
│   ├─→ CSP ResBlock ×1 (64 filters)
│   ├─→ CSP ResBlock ×2 (128 filters)
│   ├─→ CSP ResBlock ×8 (256 filters) ───→ 76×76 (用于检测小目标)
│   ├─→ CSP ResBlock ×8 (512 filters) ───→ 38×38 (用于检测中目标)
│   └─→ CSP ResBlock ×4 (1024 filters) ──→ 19×19 (用于检测大目标)
│
├─→ SPP
│   └─→ 多尺度池化
│
├─→ PANet (上采样路径)
│   ├─→ 19×19 → 38×38 → 76×76
│
├─→ PANet (下采样路径)
│   └─→ 76×76 → 38×38 → 19×19
│
└─→ YOLO Head
    ├─→ 76×76 → 3×(5+80) = 255
    ├─→ 38×38 → 3×(5+80) = 255
    └─→ 19×19 → 3×(5+80) = 255
```

---

## 二、Bag of Freebies (BoF) vs Bag of Specials (BoS)

### 2.1 概念定义

```
┌────────────────────────────────────────────────────────────────┐
│                    提升检测器性能的方法                          │
├──────────────────────────┬─────────────────────────────────────┤
│    Bag of Freebies       │         Bag of Specials             │
│    (免费午餐)             │         (特殊技巧)                   │
├──────────────────────────┼─────────────────────────────────────┤
│ • 训练时采用              │ • 增加推理成本很少                   │
│ • 推理时无成本            │ • 显著提升精度                       │
│ • 只改变训练策略或成本    │ • 修改网络结构或连接方式             │
├──────────────────────────┼─────────────────────────────────────┤
│ 数据增强:                 │ 增强感受野:                         │
│ - Mosaic                  │ - SPP, ASPP, RFB                    │
│ - MixUp, CutMix, Cutout   │                                     │
│ - Photometric distortion  │ 注意力机制:                         │
│                           │ - Squeeze-and-Excitation            │
│ 正则化:                   │ - Spatial Attention                 │
│ - DropOut, DropBlock      │                                     │
│ - Label Smoothing         │ 特征融合:                           │
│                           │ - FPN, PANet, BiFPN                 │
│ 损失函数:                 │                                     │
│ - IoU Loss, GIoU, DIoU,   │ 激活函数:                           │
│   CIoU, Focal Loss        │ - Mish, Swish                       │
│                           │                                     │
│ 样本均衡:                 │ 后处理:                             │
│ - CmBN                    │ - DIoU-NMS                          │
│ - Eliminate grid sensitivity│                                   │
└──────────────────────────┴─────────────────────────────────────┘
```

### 2.2 关键 BoF 详解

**Mosaic 数据增强：**

```
将4张图片拼接成1张：

┌─────────┬─────────┐
│  图片1   │  图片2   │
│  (左上)  │  (右上)  │
├─────────┼─────────┤
│  图片3   │  图片4   │
│  (左下)  │  (右下)  │
└─────────┴─────────┘

优点：
1. 丰富检测目标背景
2. 相当于 batch size 增加4倍
3. 减少大 batch 的 GPU 需求
4. 迫使网络学习不同尺度的物体
```

**DropBlock：**

```
传统 Dropout：随机丢弃单个神经元
DropBlock：随机丢弃连续区域（整张特征图区域）

┌─────────┐              ┌─────────┐
│ ● ○ ● ● │              │ ● ○ ○ ○ │
│ ● ● ○ ● │   Dropout    │ ○ ○ ● ● │
│ ○ ● ● ● │ ──────────→  │ ● ● ● ○ │
│ ● ● ○ ● │              │ ○ ● ● ● │
└─────────┘              └─────────┘
 随机点                    随机点

┌─────────┐              ┌─────────┐
│ ● ● ● ● │              │ ● ● ● ● │
│ ● ● ● ● │   DropBlock  │ ○ ○ ○ ● │
│ ● ● ● ● │ ──────────→  │ ○ ○ ○ ● │
│ ● ● ● ● │              │ ● ● ● ● │
└─────────┘              └─────────┘
 原图                      丢弃整片区域

更适合卷积网络：结构化信息需要结构化丢弃
```

**CIoU (Complete IoU) Loss：**

```
IoU Loss 的演进：

IoU = 交集 / 并集
    ↓ 问题：不重叠时梯度为0
GIoU = IoU - |C - (A∪B)| / |C|  (C是包围框)
    ↓ 问题：收敛慢，对中心点距离不敏感
DIoU = IoU - ρ²(b, bᵍᵗ) / c²  (中心点距离/对角线²)
    ↓ 问题：没有考虑长宽比
CIoU = IoU - ρ²/c² - αv

其中：
v = (4/π²)(arctan(wᵍᵗ/hᵍᵗ) - arctan(w/h))²  (长宽比一致性)
α = v / ((1-IoU) + v)

CIoU 同时考虑：
1. 重叠面积
2. 中心点距离
3. 长宽比一致性
```

### 2.3 关键 BoS 详解

**Mish 激活函数：**

```
f(x) = x × tanh(softplus(x))
     = x × tanh(ln(1 + eˣ))

相比 ReLU：
- 非单调（负值也有梯度）
- 平滑（处处可导）
- 自门控（self-gated）

适合深层网络，提升特征提取能力
```

**SAM (Spatial Attention Module)：**

```
通道注意力: Squeeze-and-Excitation (SE)
空间注意力: Spatial Attention Module (SAM)

SAM 结构:
输入特征图
    │
    ├──→ Conv(1×1) ──┐
    │                 ├──→ Sigmoid ──→ 空间权重图
    └──→ Conv(1×1) ──┘
    
输出 = 输入 × 空间权重图

作用：让网络关注重要的空间位置
```

---

## 三、性能对比

```
方法           Backbone      输入尺寸    FPS(V100)   mAP@0.5(COCO)
──────────────────────────────────────────────────────────────────
YOLOv3-ultra   Darknet-53    608×608     20.6        58.7%
YOLOv3-SPP     Darknet-53    608×608     20.6        60.6%
YOLOv3-tiny    Darknet-53    416×416     330         33.1%
──────────────────────────────────────────────────────────────────
YOLOv4         CSPDarknet53  608×608     62.0        65.7%
YOLOv4-tiny    CSPDarknet53  416×416     371         40.2%
──────────────────────────────────────────────────────────────────
EfficientDet-D0  EfficientNet 512×512    62.5        52.6%
EfficientDet-D1  EfficientNet 640×640    50.0        56.1%
EfficientDet-D2  EfficientNet 768×768    41.7        59.1%
EfficientDet-D3  EfficientNet 896×896    23.8        62.7%
──────────────────────────────────────────────────────────────────
RetinaNet-50   ResNet-50     750×500     61.0        59.1%
```

**结论：**
- YOLOv4: 65.7% mAP @ 62 FPS，速度精度俱佳
- YOLOv4-tiny: 40.2% mAP @ 371 FPS，超快实时检测

---

## 四、面试要点

### 4.1 高频问题

**Q1: YOLOv4 的核心贡献是什么？**

```
答：
1. 系统化总结了目标检测中的 BoF (Bag of Freebies) 和 BoS (Bag of Specials)
2. 找到了速度和精度的"最优组合"：
   - Backbone: CSPDarknet53
   - Neck: SPP + PANet
   - BoF: Mosaic, CIoU, DropBlock 等
   - BoS: Mish, SAM 等
3. 证明了通过合理的组件组合，可以在单 GPU 上快速训练出高性能检测器
```

**Q2: CSPDarknet53 相比 Darknet53 有什么优势？**

```
答：
CSP (Cross Stage Partial) 结构：
1. 将特征图分成两部分，一部分直接传递，另一部分经过残差块
2. 优势：
   - 减少重复梯度信息（约20%计算量减少）
   - 保持甚至提升精度
   - 更适合边缘设备部署
```

**Q3: 什么是 CIoU Loss？为什么比 IoU/GIoU/DIoU 好？**

```
答：
演进路线：
IoU → GIoU → DIoU → CIoU

CIoU = IoU - (中心距离²/对角线²) - αv

同时考虑三个因素：
1. 重叠面积（IoU）
2. 中心点距离（DIoU）
3. 长宽比一致性（v项）

实验证明 CIoU 收敛更快，最终精度更高
```

**Q4: Mosaic 数据增强是什么？有什么好处？**

```
答：
将4张图片随机拼接成1张：
1. 丰富背景多样性
2. 相当于 batch size 增加4倍
3. 减少大 batch 对 GPU 显存的需求
4. 迫使网络学习不同尺度的物体

YOLOv4 使用：Mosaic + MixUp + 传统增强
```

**Q5: YOLOv4 和 YOLOv3 的主要区别？**

```
答：
网络结构：
- YOLOv3: Darknet53 + FPN
- YOLOv4: CSPDarknet53 + SPP + PANet

训练技巧：
- YOLOv4 引入大量 BoF (Mosaic, DropBlock, CIoU, Label Smoothing)

激活函数：
- YOLOv3: Leaky ReLU
- YOLOv4: Mish

性能：
- YOLOv4 在相同速度下 mAP 提升约 8%
```

### 4.2 手撕代码要点

```python
# CIoU Loss 实现
def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    pred_boxes: [x, y, w, h]
    target_boxes: [x, y, w, h]
    """
    # 转换为 (x1, y1, x2, y2)
    pred_x1, pred_y1 = pred_boxes[:, 0] - pred_boxes[:, 2]/2, pred_boxes[:, 1] - pred_boxes[:, 3]/2
    pred_x2, pred_y2 = pred_boxes[:, 0] + pred_boxes[:, 2]/2, pred_boxes[:, 1] + pred_boxes[:, 3]/2
    
    target_x1, target_y1 = target_boxes[:, 0] - target_boxes[:, 2]/2, target_boxes[:, 1] - target_boxes[:, 3]/2
    target_x2, target_y2 = target_boxes[:, 0] + target_boxes[:, 2]/2, target_boxes[:, 1] + target_boxes[:, 3]/2
    
    # 交集
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 并集
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area + eps
    
    # IoU
    iou = inter_area / union_area
    
    # 中心点距离
    center_dist_sq = (pred_boxes[:, 0] - target_boxes[:, 0])**2 + (pred_boxes[:, 1] - target_boxes[:, 1])**2
    
    # 包围框对角线
    c_x1, c_y1 = torch.min(pred_x1, target_x1), torch.min(pred_y1, target_y1)
    c_x2, c_y2 = torch.max(pred_x2, target_x2), torch.max(pred_y2, target_y2)
    c_diag_sq = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2 + eps
    
    # 长宽比一致性
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(target_boxes[:, 2] / target_boxes[:, 3]) - 
        torch.atan(pred_boxes[:, 2] / pred_boxes[:, 3]), 2)
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    
    # CIoU
    ciou = iou - center_dist_sq / c_diag_sq - alpha * v
    loss = 1 - ciou
    
    return loss.mean()

# Mish 激活函数
def mish(x):
    return x * torch.tanh(F.softplus(x))
```

---

## 五、总结

| 特性 | YOLOv4 贡献 |
|:---:|:---|
| **系统总结** | 提出 BoF 和 BoS 概念，分类整理检测技术 |
| **最优组合** | CSPDarknet53 + SPP + PANet + 各种训练技巧 |
| **速度精度** | 65.7% mAP @ 62 FPS，最优平衡点 |
| **工程价值** | 单 GPU 可训练，易于复现和部署 |

**一句话：** YOLOv4 通过大量的消融实验，找到了目标检测器的"最优超参数组合"，证明了合理的组件选择和训练技巧可以在单 GPU 上达到最先进的速度-精度平衡。

---

## 参考资源

- 论文: https://arxiv.org/abs/2004.10934
- 代码: https://github.com/AlexeyAB/darknet
- 中文解读: https://zhuanlan.zhihu.com/p/136172670
