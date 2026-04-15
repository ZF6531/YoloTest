# YOLOv7

> 论文: "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors" (CVPR 2023)
>
> 作者: Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao

---

## 一、前言

YOLOv7 由 YOLOv4 的原班人马打造，专注于**训练策略优化**（Bag of Freebies），在不增加推理成本的前提下提升精度。

**核心定位：**
- 可训练的免费技巧（Trainable BoF）
- 实时检测的新 SOTA
- 5-160 FPS 范围内的最佳速度-精度平衡

---

## 二、网络架构

### 2.1 整体结构

```
输入 (640×640×3)
    ↓
┌─────────────────────────────────────────┐
│  Backbone: E-ELAN (Extended ELAN)       │
│  - 扩展的高效层聚合网络                  │
│  - 计划性重参数化卷积                    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Neck: SPPCSPC + ELAN + MPConv          │
│  - 改进的空间金字塔池化                  │
│  - 尺度感知特征融合                      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Head: RepConv + 动态标签分配            │
│  - 重参数化检测头                        │
│  - 粗到细的指导                          │
└─────────────────────────────────────────┘
```

### 2.2 E-ELAN 骨干网络

**ELAN (Efficient Layer Aggregation Network)：**

```
核心思想：控制最短最长梯度路径，使网络学习到更多样特征

传统堆叠：                        ELAN：
输入                               输入
  │                                  │
  ↓                                  ↓
Conv ──→ Conv ──→ Conv ──→ 输出      ├──→ Conv ──┐
                                          │        ├──→ Concat ──→ Conv ──→ 输出
                                          ↓        │
                                    ├───────────┐  │
                                    │ Conv Block│──┘
                                    │ ×N        │
                                    └───────────┘

扩展版 E-ELAN：
- 使用 expand、shuffle、merge 基数
- 不破坏原有梯度路径
- 增强网络学习能力
```

### 2.3 模型缩放（Model Scaling）

YOLOv7 提供了系统化的模型缩放：

| 模型 | 宽度 | 深度 | 参数量 | mAP | FPS (V100) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| YOLOv7 | 1.0 | 1.0 | 36.9M | 51.2% | 161 |
| YOLOv7-X | 1.25 | 1.58 | 71.3M | 52.9% | 103 |
| YOLOv7-W6 | 1.25 | 1.0 | 67.5M | 54.2% | 86 |
| YOLOv7-E6 | 1.25 | 1.31 | 97.2M | 55.2% | 73 |
| YOLOv7-D6 | 1.25 | 1.62 | 154.7M | 56.1% | 56 |
| YOLOv7-E6E | 1.25 | 2.94 | 151.7M | 56.8% | 45 |

---

## 三、关键创新：可训练的 Bag of Freebies

### 3.1 计划性重参数化 (Planned Reparameterization)

```
问题：在哪些层使用重参数化？

研究发现：
- 在直连层（如 ResNet 中的 identity）使用 RepConv 会破坏残差连接
- 在块内使用 RepConv 效果更好

策略：
1. 仅在块的内部使用 RepConv
2. 有残差连接的地方不使用重参数化
3. 计划性地替换卷积层
```

### 3.2 粗到细引导的辅助头 (Coarse-to-Fine Lead Head)

```
YOLOv7 使用双头设计：

Lead Head (主导头)：
- 负责最终预测
- 使用标准标签分配

Auxiliary Head (辅助头)：
- 训练时使用，推理时丢弃
- 使用粗标签（lead head 引导）
- 帮助网络学习

粗到细过程：
1. Lead head 产生软标签（soft label）
2. Auxiliary head 学习这些软标签
3. 从粗粒度到细粒度逐步优化

优势：
- 不增加推理成本
- 提升训练效果
- 类似知识蒸馏，但是自我蒸馏
```

### 3.3 动态标签分配

```
YOLOv7 结合了多种标签分配策略：

1. 训练初期：
   - 使用 auxiliary head
   - 更宽松的标签分配
   - 让更多样本参与学习

2. 训练后期：
   - 使用 lead head
   - 更精确的标签分配
   - 专注于高质量样本

3. 动态调整：
   - 根据训练进度调整分配策略
   - 渐进式学习
```

### 3.4 其他 BoF 技巧

```
1. 卷积重参数化 (RepConv)
   - 训练：多分支结构
   - 推理：融合为单分支
   
2. EMA (Exponential Moving Average)
   - 模型参数的滑动平均
   - 提升模型稳定性

3. IoU 感知损失
   - 考虑预测框与 GT 的 IoU
   - 更精确的边界框学习

4. Mosaic + MixUp 增强
   - 组合数据增强
   - 提升泛化能力
```

---

## 四、性能对比

### 4.1 与竞品对比（COCO 数据集）

```
模型              输入    参数量   mAP    FPS(V100)  FPS(batch=32)
─────────────────────────────────────────────────────────────────────
YOLOv5-N (r6.1)  640    1.9M     28.0%   602        4237
YOLOv7-tiny      640    6.2M     35.2%   952        5265
─────────────────────────────────────────────────────────────────────
YOLOv5-S (r6.1)  640    7.2M     37.4%   224        1615
YOLOv7-tiny      640    6.2M     35.2%   952        5265
YOLOv7           640    36.9M    51.2%   161        730
─────────────────────────────────────────────────────────────────────
YOLOv5-M (r6.1)  640    21.2M    45.4%   130        889
YOLOv7-X         640    71.3M    52.9%   103        468
─────────────────────────────────────────────────────────────────────
YOLOv5-L (r6.1)  640    46.5M    49.0%   83         551
YOLOv7-E6        1280   97.2M    55.9%   50         256
─────────────────────────────────────────────────────────────────────
YOLOv5-X (r6.1)  640    86.7M    50.7%   62         365
YOLOv7-D6        1280   154.7M   56.6%   39         192
─────────────────────────────────────────────────────────────────────
```

### 4.2 关键结论

```
1. YOLOv7-tiny：
   - 比 YOLOv5-N 快 1.6 倍
   - mAP 高 7.2%

2. YOLOv7：
   - 比 YOLOv5-L 快 1.9 倍
   - mAP 高 2.2%

3. YOLOv7-X：
   - 比 YOLOv5-X 快 1.7 倍
   - mAP 高 2.2%

结论：YOLOv7 在 5-160 FPS 范围内全面超越竞品
```

---

## 五、部署优化

### 5.1 推理优化

```
YOLOv7 支持多种推理框架：

1. PyTorch
2. ONNX
3. TensorRT
4. OpenVINO
5. CoreML
6. PaddlePaddle

TensorRT 优化：
- FP16 推理
- INT8 量化
- 动态批次
```

### 5.2 模型压缩

```
YOLOv7 提供了剪枝和量化支持：

1. 通道剪枝
   - 移除不重要的通道
   - 减少参数量和计算量

2. 层剪枝
   - 移除冗余层

3. 量化
   - FP32 → FP16/INT8
   - 速度提升 2-4 倍
```

---

## 六、面试要点

### 6.1 高频问题

**Q1: YOLOv7 的核心创新是什么？**

```
答：
1. E-ELAN 骨干网络：扩展的高效层聚合网络
2. 计划性重参数化：在合适的位置使用 RepConv
3. 粗到细引导的辅助头：训练时辅助，推理时丢弃
4. 动态标签分配：根据训练进度调整策略
5. 大量可训练的 BoF：不增加推理成本，提升精度
```

**Q2: 什么是粗到细引导的辅助头？**

```
答：
YOLOv7 使用双头设计：

Lead Head（主导头）：
- 负责最终预测
- 使用精确标签

Auxiliary Head（辅助头）：
- 仅训练时使用
- 使用 lead head 产生的软标签
- 粗粒度学习 → 细粒度优化

优势：
- 不增加推理成本
- 类似自蒸馏效果
- 提升训练稳定性
```

**Q3: YOLOv7 为什么比 YOLOv5 快？**

```
答：
1. 更高效的 E-ELAN 结构
2. 计划性的重参数化
3. 更好的训练策略收敛更快
4. 针对 V100 等硬件优化

具体数据：
- YOLOv7 vs YOLOv5-L：快 1.9 倍，精度高 2.2%
- YOLOv7-tiny vs YOLOv5-N：快 1.6 倍，精度高 7.2%
```

**Q4: E-ELAN 是什么？**

```
答：
E-ELAN (Extended Efficient Layer Aggregation Network)：

核心思想：
- 控制最短和最长梯度路径
- 使用 expand、shuffle、merge 基数
- 在不破坏原有梯度路径的前提下增强学习能力

结构特点：
- 多分支并行
- 梯度聚合
- 特征复用
```

**Q5: 计划性重参数化和 YOLOv6 的 RepVGG 有什么区别？**

```
答：
YOLOv6 RepVGG：
- 所有块都使用重参数化
- 训练多分支，推理单分支

YOLOv7 计划性重参数化：
- 只在块内部使用重参数化
- 有残差连接的地方不用
- 避免破坏残差连接
- 更精细的设计
```

### 6.2 手撕代码要点

```python
# ELAN 模块
class ELAN(nn.Module):
    def __init__(self, c1, c2, n=4):
        super().__init__()
        c_ = c1 // 2
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(c_, c_, 3, 1)
        self.conv4 = Conv(c_, c_, 3, 1)
        # ... 多个卷积块
        self.concat_conv = Conv(c_ * n, c2, 1, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        # 聚合多个分支
        x = torch.cat([x1, x2], dim=1)
        return self.concat_conv(x)

# RepConv（重参数化卷积）
class RepConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, padding=1, groups=1):
        super().__init__()
        # 训练时的多分支
        self.conv_kxk = nn.Conv2d(c1, c2, k, s, padding, groups=groups)
        self.conv_1x1 = nn.Conv2d(c1, c2, 1, s, 0, groups=groups)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    
    def forward(self, x):
        if self.training:
            return self.act(self.bn(self.conv_kxk(x) + self.conv_1x1(x)))
        else:
            # 推理时融合
            return self.act(self.rbr_reparam(x))
    
    def switch_to_deploy(self):
        # 重参数化融合
        pass

# 辅助头 + 主导头
def dual_head_loss(aux_pred, lead_pred, targets):
    """
    辅助头使用主导头产生的软标签
    """
    # Lead head 损失
    lead_loss = compute_loss(lead_pred, targets)
    
    # Auxiliary head 使用 lead head 的软标签
    soft_labels = lead_pred.detach()
    aux_loss = compute_loss(aux_pred, soft_labels)
    
    return lead_loss + 0.25 * aux_loss  # 辅助头权重较小
```

---

## 七、总结

| 特性 | YOLOv7 特点 |
|:---:|:---|
| **E-ELAN** | 扩展高效层聚合网络，控制梯度路径 |
| **计划性重参数化** | 精细设计重参数化位置 |
| **双头设计** | Lead + Auxiliary，粗到细学习 |
| **动态标签分配** | 根据训练进度调整策略 |
| **BoF** | 大量可训练技巧，不增加推理成本 |
| **性能** | 5-160 FPS 范围内的 SOTA |

**一句话：** YOLOv7 通过 E-ELAN 架构、计划性重参数化和粗到细的辅助头设计，在不增加推理成本的前提下，实现了实时检测的新 SOTA。

---

## 参考资源

- GitHub: https://github.com/WongKinYiu/yolov7
- 论文: https://arxiv.org/abs/2207.02696
- 中文解读: https://zhuanlan.zhihu.com/p/543184098
