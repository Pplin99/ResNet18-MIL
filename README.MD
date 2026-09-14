# ResNet-18 多实例学习的甲状腺超声患者级分类

本项目面向一位患者包含多张甲状腺超声图像的分类任务，使用 ResNet-18 提取图像特征，并通过门控注意力多实例学习（Gated-Attention MIL）将多张图像的信息聚合为患者级预测。项目包含数据检查、模型训练、优化器比较和结果分析程序，主要通过 Python 命令行运行。

## 实验目的

实验旨在利用同一患者的多幅超声图像建立患者级二分类模型，研究注意力聚合在多图像建模中的应用，并比较 AdamW 与 SGD 两种优化器配置下的训练表现。

监督标签来自患者信息表中的“颈部淋巴结”字段，模型输出对应这一标签的阳性概率。该标签不等同于病理证实的淋巴结转移，结果解释应以原始标签定义为准。

实验采用患者级数据划分，同一患者的全部图像归入同一个集合，以避免不同集合之间出现患者信息泄漏。模型和优化器配置依据验证集表现选择，测试集用于最终评价。

## 模型与训练方法

每位患者对应一个图像袋。袋内每张图像经过共享的 ResNet 编码器，得到图像特征；门控注意力模块为不同图像分配权重，再对特征加权汇总，最后通过分类层输出患者级概率。

```text
患者的多张超声图像 → 共享 ResNet 编码器 → 门控注意力聚合 → 患者级特征 → 二分类概率
```

图像预处理采用等比例缩放与黑色填充，将输入统一为 224 × 224，并使用 ImageNet 均值和标准差归一化。训练阶段增加小角度随机旋转、亮度和对比度扰动。每位患者每轮最多随机采样 16 张图像，验证与测试默认使用该患者的全部图像。

主要实验配置如下：

| 配置项 | 设置 |
|---|---|
| 实验骨干 | ResNet-18，支持切换为 ResNet-50 |
| 初始化 | ImageNet 预训练权重 |
| 随机种子 | 20260811 |
| 注意力隐藏维度 | 256 |
| Dropout | 0.30 |
| 损失函数 | 带阳性类别权重的 BCEWithLogitsLoss，权重为 145 / 29 = 5 |
| AdamW 学习率 | 1e-4 |
| SGD 学习率 | 1e-3，momentum=0.9，启用 Nesterov |
| 权重衰减 | 1e-4 |
| 学习率调度 | 余弦退火 |
| 最大训练轮数 | 120 |
| 主干冻结 | 前 3 轮，之后解冻 |
| 梯度累积 | 每 4 位患者更新一次，末尾不足 4 位时也更新 |
| 早停 | 验证集 AP 连续 15 轮未超过已保存最佳值 0.001 |
| 计算方式 | CUDA GPU，自动混合精度 |

两组实验保持相同的数据划分、初始化种子及其他训练设置，分别采用上表中的优化器和对应学习率。代码的骨干默认值为 `resnet50`，运行本项目的 ResNet-18 实验时需显式指定 `--backbone resnet18`。

## 环境搭建

使用 Python 3.11。主要计算依赖为 PyTorch 2.5.1、torchvision 0.20.1、NumPy、Pillow 和 scikit-learn；Matplotlib 用于结果绘图。完整训练需要支持 CUDA 的 NVIDIA GPU。

在项目根目录创建并激活环境：

```bash
python -m venv .venv
```

Linux 使用 `source .venv/bin/activate`；Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1`。激活后安装计算依赖，以下以 CUDA 12.1 构建为例：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==2.4.2 Pillow==12.1.1 scikit-learn==1.8.0 matplotlib==3.10.8
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

运行训练前应确认 `CUDA available` 为 `True`。其他 CUDA 构建可根据驱动兼容性参照 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v251)选择。

项目中的 `requirements.txt` 固定的是 CPU 示例与可视化环境，`requirements-lock.txt` 记录该环境在 Windows / Python 3.11 下的依赖版本。仅运行 CPU 示例时，可在独立环境中执行：

```bash
python -m pip install -r requirements.txt -c requirements-lock.txt
```

该依赖文件包含 CPU 版 PyTorch，不适用于直接配置完整训练环境。训练、数据验证和模型比较脚本均可独立运行；`results.ipynb` 用于辅助查看示例图表。

## 数据结构

实验使用私有甲状腺超声数据，共 290 位患者、3,706 张图像，阳性患者 49 位、阴性患者 241 位，不使用 TN3K 数据。数据按患者划分为训练、验证和测试集：

| 集合 | 患者数 | 阳性 | 阴性 |
|---|---:|---:|---:|
| 训练集 | 174 | 29 | 145 |
| 验证集 | 58 | 10 | 48 |
| 测试集 | 58 | 10 | 48 |
| 合计 | 290 | 49 | 241 |

数据需另行准备，预期目录如下：

```text
data/private_mil/
├── patient_info.csv
├── patient_manifest.csv
├── image_manifest.csv
├── split_summary.json
└── images/
    ├── train/<patient_id>/<image_file>
    ├── val/<patient_id>/<image_file>
    └── test/<patient_id>/<image_file>
```

两个清单文件为程序的主要数据入口，其必需字段为：

| 文件 | 字段 | 含义 |
|---|---|---|
| patient_manifest.csv | patient_id | 患者编号，当前验证程序要求为 1–290 |
| patient_manifest.csv | split | train、val 或 test |
| patient_manifest.csv | label | 二分类标签，0 或 1 |
| patient_manifest.csv | image_count | 该患者的图像数量 |
| image_manifest.csv | patient_id | 图像所属患者编号 |
| image_manifest.csv | split | 所属集合，需与患者清单一致 |
| image_manifest.csv | label | 标签，需与患者清单一致 |
| image_manifest.csv | image_relpath | 相对于 data/private_mil 的图像路径 |

`patient_info.csv` 和 `split_summary.json` 用于保留数据整理信息，训练入口直接读取 `image_manifest.csv`。同一患者的所有图像应具有相同标签与集合归属。当前校验程序和训练入口包含上述固定样本数量检查，替换数据集时需同步调整。

## 代码包说明

```text
.
├── code/
│   ├── train_resnet_mil.py
│   ├── validate_package.py
│   ├── compare_models.py
│   └── reproduce_demo.py
├── slurm/
│   ├── train_optimizer.slurm
│   └── submit_two_optimizers.sh
├── results.ipynb
├── requirements.txt
├── requirements-lock.txt
├── README_服务器训练.md
└── README.MD
```

| 文件 | 功能 |
|---|---|
| code/train_resnet_mil.py | 患者图像袋加载、预处理、ResNet 与注意力模型、训练、早停及预测导出 |
| code/validate_package.py | 检查患者与图像数量、标签和划分一致性、文件完整性，并抽样验证图像 |
| code/compare_models.py | 汇总 AdamW 与 SGD 的评价指标，按验证集 AP 选择优化器 |
| code/reproduce_demo.py | 使用合成图像运行小规模 CPU 示例，检查模型计算流程 |
| slurm/train_optimizer.slurm | 在 Slurm 集群运行单个优化器的训练任务 |
| slurm/submit_two_optimizers.sh | 分别提交 AdamW 与 SGD 任务并保存任务编号 |
| results.ipynb | 展示合成示例的输入、损失、预测和注意力图表 |
| README_服务器训练.md | 服务器训练环境与调度操作的补充说明 |

## 运行流程

### 1. 检查数据

```bash
python code/validate_package.py --data-root data/private_mil --decode-sample 30
```

程序检查清单与实际文件的一致性，并输出样本统计和图像清单的 SHA-256 校验值。

### 2. 训练模型

在项目根目录分别执行：

```bash
python code/train_resnet_mil.py --data-root data/private_mil --output-dir runs/resnet18_adamw --optimizer adamw --backbone resnet18 --seed 20260811 --weights imagenet
python code/train_resnet_mil.py --data-root data/private_mil --output-dir runs/resnet18_sgd --optimizer sgd --backbone resnet18 --seed 20260811 --weights imagenet
```

输出目录必须尚不存在；重复实验时应使用新的目录名。首次加载 ImageNet 权重需要下载，也可将 `TORCH_HOME` 环境变量设置为已有权重缓存目录，例如项目的 `.cache/torch` 绝对路径。

若在 Slurm 集群运行，先按实际环境修改脚本中的项目路径、Python 解释器、分区和日志路径，再执行：

```bash
BACKBONE=resnet18 SEED=20260811 WEIGHTS=imagenet bash slurm/submit_two_optimizers.sh
```

### 3. 比较优化器

```bash
python code/compare_models.py --adamw runs/resnet18_adamw --sgd runs/resnet18_sgd --output outputs/comparison.json
```

比较程序读取两组训练的 `summary_metrics.json`，汇总验证集与测试集指标，并以验证集 AP 较高的配置作为选择结果。

## 结果与评价

每次完整训练会生成以下文件：

| 输出文件 | 内容 |
|---|---|
| config.json | 训练参数、运行设备及 PyTorch 版本 |
| best.pt | 按验证集 AP 与最小改进量规则保存的模型参数 |
| history.csv | 每轮训练损失、验证损失、验证指标及学习率 |
| summary_metrics.json | 最佳轮次、验证集和测试集指标、运行时间 |
| val_predictions.csv / test_predictions.csv | 患者编号、标签、预测概率、预测类别及阈值 |
| val_attention.csv / test_attention.csv | 各患者每张图像的注意力权重 |

模型评价包括 ROC-AUC、Average Precision（AP）、准确率、精确率、敏感度、特异度、F1、Brier 分数和混淆矩阵。代码中的 `pr_auc` 字段实际使用 `average_precision_score` 计算，表示 AP。分类阈值由验证集上的 Youden 指数确定，再应用于测试集。

当前项目未附带私有数据的完整训练结果，暂不列出真实实验的性能数值或两种优化器的优劣结论。完成训练后，可依据上述结果文件分析收敛情况、分类性能和图像注意力分布。

项目另提供一个可独立运行的合成数据示例：

```bash
python code/reproduce_demo.py --output-dir outputs/demo
```

该示例冻结随机初始化的 ResNet-18 编码器，训练注意力模块和分类头，共运行 6 轮。在已验证的 CPU 环境中，训练损失由约 0.7212 降至 0.5891，相同种子下重复运行的输出一致。结果保存为 `outputs/demo/demo_results.json`，配套图表见 [results.ipynb](results.ipynb)。这些数值仅用于说明程序运行情况，不代表真实超声数据上的分类性能。
