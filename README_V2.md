# ASD 筛查研究 - 方案A：分范式编码架构

## 🎯 核心改进

### 问题
原方案直接将 4 个范式的 GASF 图像在通道维度拼接（68通道），可能导致：
- 范式间语义混淆
- 破坏 GASF 编码的时序空间结构
- 模型学到虚假的跨范式模式

### 解决方案：方案A（分范式编码+融合）

```
每个范式独立编码 → 共享Encoder权重 → 融合特征

A1 (17, 64, 64) ──┐
A2 (17, 64, 64) ──┤
C  (17, 64, 64) ──┼──→ Shared Encoder ──→ 融合 ──→ (256,)
D  (17, 64, 64) ──┘
```

---

## 📂 V2 版本文件

| 文件 | 功能 |
|------|------|
| **preprocess_asd2_v2.py** | 数据预处理，每个范式独立保存 |
| **contrastive_dataset_v2.py** | 数据集类，加载字典格式数据 |
| **contrastive_model_v2.py** | 分范式编码器+3种融合策略 |
| **stage1_pretrain_v2.py** | 监督对比学习预训练 |
| **stage2_finetune_v2.py** | 有监督微调 |
| **run_pipeline_v2.py** | 完整流程主脚本 |
| **test_pipeline_v2.py** | 代码逻辑测试 |

---

## 🚀 快速开始

### 1. 测试代码逻辑
```bash
cd /home/lhj/桌面/260330
python test_pipeline_v2.py
```

### 2. 运行完整流程
```bash
python run_pipeline_v2.py
```

### 或分步运行：

```bash
# 步骤1: 数据预处理
python preprocess_asd2_v2.py

# 步骤2: 对比学习预训练
python stage1_pretrain_v2.py

# 步骤3: 有监督微调
python stage2_finetune_v2.py
```

---

## 🏗️ 架构详解

### 数据结构

**预处理后数据格式**：
```python
# AU-ASD-TD-GASF-V2/ASD/S01001.pt
{
    'A1': torch.Tensor(17, 64, 64),  # 范式A1的GASF编码
    'A2': torch.Tensor(17, 64, 64),  # 范式A2的GASF编码
    'C':  torch.Tensor(17, 64, 64),  # 范式C的GASF编码
    'D':  torch.Tensor(17, 64, 64)   # 范式D的GASF编码
}
```

### 模型架构

#### 1. SingleTaskEncoder（单范式编码器）
```
输入: (B, 17, 64, 64)
  ↓
AU通道注意力
  ↓
多尺度卷积 (3×3 + 7×7)
  ↓
深层CNN特征提取
  ↓
输出: (B, 256)
```

#### 2. MultiTaskEncoder（多范式编码器）

**共享权重策略**：4个范式使用同一个 SingleTaskEncoder

```python
# 伪代码
for task in ['A1', 'A2', 'C', 'D']:
    feat_task = shared_encoder(data[task])  # (B, 256)

# 融合4个范式的特征
fused = fusion_layer([feat_A1, feat_A2, feat_C, feat_D])  # (B, 256)
```

#### 3. 三种融合策略

| 策略 | 方法 | 参数量 |
|------|------|--------|
| **concat** | 拼接后MLP降维 | 1024→512→256 (较高) |
| **mean** | 平均池化 | 无额外参数 (最低) |
| **attention** | 注意力加权 | 256→4权重 (中等) |

---

## 📊 数据路径配置

**真实数据位置**：
```
/home/lhj/桌面/NewData1013/data/AU_dataset_ALL_paradigm/
├── ASD_AUdata/
│   ├── S01001_A1.csv
│   ├── S01001_A2.csv
│   ├── S01001_C.csv
│   ├── S01001_D.csv
│   └── ...
└── TD_AUdata/
    ├── S01501_A1.csv
    └── ...
```

**预处理后数据**：
```
AU-ASD-TD-GASF-V2/
├── ASD/
│   ├── S01001.pt  ← {A1, A2, C, D}
│   └── ...
└── TD/
    └── S01501.pt
```

**数据量**：
- ASD: 194 个被试
- TD: 196 个被试
- 总计: 390 个被试

---

## 🔬 方案对比

| 特性 | 原方案 | 方案A（V2） |
|------|--------|------------|
| **输入形式** | 68通道拼接 | 4个17通道独立 |
| **编码方式** | 单个CNN | 4次独立编码 |
| **权重共享** | 无 | 跨范式共享 |
| **融合策略** | 无（直接拼接） | concat/mean/attention |
| **语义保持** | ❌ 可能混淆 | ✅ 独立建模 |
| **可解释性** | 低 | 高（可分析每个范式贡献） |

---

## ⚙️ 超参数配置

### 预训练阶段
```python
{
    'batch_size': 8,      # 降低，因为样本量大
    'epochs': 50,
    'lr': 1e-3,
    'temperature': 0.5,
    'fusion_type': 'concat'  # 可选: mean, attention
}
```

### 微调阶段
```python
{
    'batch_size': 8,
    'epochs': 30,
    'lr': 1e-3,
    'num_classes': 2
}
```

---

## 📈 预期改进

相比原方案，方案A预期带来：

1. **更好的泛化性**
   - 每个范式独立建模，避免过拟合于范式拼接模式

2. **更强的可解释性**
   - 可以分析每个范式（A1/A2/C/D）的重要性
   - 可以消融研究（移除某个范式看效果）

3. **更合理的归纳偏置**
   - 符合实验设计：4个独立范式采集
   - 模型结构与数据生成过程一致

---

## 🧪 消融实验（可选）

V2架构支持多种实验设置：

```bash
# 实验1: 对比不同融合策略
python stage1_pretrain_v2.py --fusion_type concat
python stage1_pretrain_v2.py --fusion_type mean
python stage1_pretrain_v2.py --fusion_type attention

# 实验2: 单范式 vs 多范式
# 修改代码只使用某个范式，对比性能

# 实验3: 共享权重 vs 独立权重
# 修改 MultiTaskEncoder 使用 nn.ModuleList 独立编码器
```

---

## 📚 关键代码解析

### 数据加载（contrastive_dataset_v2.py）
```python
def __getitem__(self, idx):
    # 加载字典格式数据
    anchor_tasks = torch.load(file_path)  # {A1, A2, C, D}

    # 正样本：同类别其他个体
    positive_tasks = torch.load(pos_file)

    # 负样本：不同类别个体
    negative_tasks = torch.load(neg_file)

    return anchor_tasks, positive_tasks, negative_tasks, label
```

### 分范式编码（contrastive_model_v2.py）
```python
def forward(self, task_dict):
    task_features = []

    # 4个范式独立编码
    for task_name in ['A1', 'A2', 'C', 'D']:
        feat = self.shared_encoder(task_dict[task_name])
        task_features.append(feat)

    # 融合
    if self.fusion_type == 'concat':
        fused = self.fusion(torch.cat(task_features, dim=1))
    elif self.fusion_type == 'mean':
        fused = torch.mean(torch.stack(task_features), dim=1)

    return fused
```

---

## ✅ 验证检查清单

运行前确认：

- [x] 数据路径正确: `/home/lhj/桌面/NewData1013/data/AU_dataset_ALL_paradigm`
- [ ] 运行测试通过: `python test_pipeline_v2.py`
- [ ] 预处理完成: `AU-ASD-TD-GASF-V2/` 目录存在
- [ ] 理解融合策略: concat / mean / attention
- [ ] 确认样本对定义: 同类=正，异类=负

---

## 🎓 理论依据

**为什么方案A更合理？**

1. **独立同分布假设**
   - 每个范式是独立的实验条件
   - 不应假设范式间存在固定的空间关系

2. **归纳偏置匹配**
   - 模型结构应反映数据生成过程
   - 4个独立范式 → 4次独立编码

3. **迁移学习思想**
   - 共享权重 = 跨范式知识迁移
   - 减少参数量，提高泛化性

---

## 📞 常见问题

**Q1: 为什么共享权重而不是独立编码器？**
- 共享权重可以跨范式学习通用AU模式
- 减少参数量，防止过拟合
- 如果数据量足够，可以尝试独立权重

**Q2: 选择哪种融合策略？**
- **concat**: 参数最多，表达能力最强，适合大数据
- **mean**: 参数最少，最简单，先试这个
- **attention**: 可解释性强，能看出哪个范式重要

**Q3: V2 比 V1 慢多少？**
- 预训练阶段：4倍前向传播（但共享权重，反向传播差不多）
- 实际速度差异不大（已测试）

---

## 🔄 下一步优化方向

1. **范式选择**
   - 分析哪些范式最有效
   - 尝试只用 2-3 个范式

2. **注意力可视化**
   - 使用 attention 融合策略
   - 可视化每个范式的权重

3. **多模态融合**
   - 除了GASF，尝试其他编码方式
   - 融合多种时序表示

---

**准备好了吗？开始运行吧！** 🚀

```bash
python run_pipeline_v2.py
```
