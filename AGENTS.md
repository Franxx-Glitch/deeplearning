# AGENTS.md — 深度学习图像分类识别 项目指南

## 项目概述

- **课程**：清华大学深度学习课程 — 实验1
- **任务**：基于 CIFAR-10 数据集，纯 NumPy 从零实现 Logistic 回归（二分类）和 SoftMax 回归（十分类）
- **禁止使用**：TensorFlow、PyTorch 等深度学习框架
- **数据**：CIFAR-10（50,000 训练 / 10,000 测试，32×32 RGB → 3072 维向量）

## 环境与命令

| 项目 | 值 |
|------|-----|
| 语言 | Python 3.14 |
| 依赖 | `numpy` `matplotlib` `tqdm` `jupyter` |
| 安装依赖 | `pip install numpy matplotlib tqdm jupyter` |
| 运行 notebook | `python -m jupyter nbconvert --to notebook --execute --output <输出名>.ipynb <输入>.ipynb` |

## 项目文件结构

```
├── hw1a.ipynb                  # ★ Logistic 回归（二分类：airplane vs automobile）
├── hw1b.ipynb                  # ★ SoftMax 回归（十分类，全部 10 类）
├── hw1a_no_bias.ipynb          # 对照：无 bias 的 Logistic
├── hw1a_momentum.ipynb         # 对照：有 momentum 的 Logistic
├── hw1b_no_bias.ipynb          # 对照：无 bias 的 SoftMax
├── hw1b_momentum.ipynb         # 对照：有 momentum 的 SoftMax
├── hw1b_bs{50,200,500}.ipynb   # 对照：不同 batch size（可变 step 数）
├── hw1b_fixedsteps_bs*.ipynb   # 对照：固定 step=500 下不同 batch size
├── *_executed.ipynb            # 上述 notebook 的运行输出（生成文件，勿手动编辑）
├── data/
│   └── cifar-10-python.zip     # CIFAR-10 数据集（164MB，需自行准备）
├── 实验1报告.docx               # 完整实验报告（含表格和曲线图）
├── 实验1要求.html               # 原始作业要求
└── .gitignore                  # 排除数据/运行输出/临时文件
```

## 核心超参数

| 参数 | 值 |
|------|-----|
| learning_rate | 0.001 |
| momentum | 0.9 |
| max_epoch | 10 |
| batch_size | 100（默认） |

## 模型实现要点

### hw1a — Logistic 回归
- 权重 `W`: `(3072,)`, 偏置 `b`: 标量
- 激活：Sigmoid `1/(1+e⁻ᶻ)`
- 损失：二元交叉熵
- 标签：`(y_true != active_label_indices[0])` 转 0/1

### hw1b — SoftMax 回归
- 权重 `W`: `(3072, 10)`, 偏置 `b`: `(10,)`
- 激活：数值稳定 Softmax（减 max 防溢出）
- 损失：交叉熵
- 标签：`np.eye(10)[y_true]` 转 one-hot

### 通用
- 普通 SGD 更新：`W -= lr * grad_W`
- 动量版本需添加 `v_W = np.zeros_like(W)` 并改为 `v_W = momentum * v_W + lr * grad_W; W -= v_W`
- log 计算加 `1e-8` 防 log(0)
- 梯度计算禁止 for 循环，必须向量化

## 实验结果摘要

| 实验 | Logistic (二分类) | SoftMax (十分类) |
|------|-------------------|-------------------|
| 基线（有 bias, 无 momentum） | 78.5% | 36.0% |
| 无 bias | 78.3%（−0.2%） | 35.9%（−0.1%） |
| 有 momentum | 81.4%（+2.9%） | 38.5%（+2.5%） |

- **bias 影响极小**（<0.3%），可省略
- **momentum 提升显著**（+2.5~2.9%），且加速收敛
- **batch size**：固定 step 数后，不同 bs 结果几乎一致（36.2~36.6%）；性能差异真正来源于每 epoch 的更新次数

## 禁止与注意事项

- **不可编辑** `*_executed.ipynb` — 这些是运行生成的输出文件
- **不可编辑** `data/` 下数据文件 — 太大且不纳入版本控制
- **不可使用** TensorFlow / PyTorch / Keras 等框架
- **不可使用** for 循环逐一处理标量，必须向量化
- 数据文件需自行从 CIFAR-10 官网下载 zip 格式放入 `data/` 目录
- notebook 运行依赖 `data/cifar-10-python.zip` 存在且格式正确

## Git 信息

- 仓库：[Franxx-Glitch/deeplearning](https://github.com/Franxx-Glitch/deeplearning)
- 分支：`master`
- 代理：`http://127.0.0.1:10793`（如推送失败需配置 `git config http.proxy`）
- `.gitignore` 已配置：排除数据文件、运行输出、临时图片、Python 缓存
