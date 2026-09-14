# 私有甲状腺超声患者级 ResNet-50 训练包

本项目只使用私有数据，不读取 TN3K。当前监督标签来自 `病人信息.csv` 的 `颈部淋巴结`，因此模型输出是该字段的患者级概率，不应直接解释为病理证实的淋巴结转移概率。

## 数据与切分

- 患者：290（阳性49、阴性241）
- 图像：3,706张
- 固定随机种子：`20260811`
- 训练：174人（阳性29、阴性145）
- 验证：58人（阳性10、阴性48）
- 测试：58人（阳性10、阴性48）
- 所有图像严格按患者划分，不存在同一患者跨集合。

服务器结构：

```text
/appsnew/home/jlpan/ResNet/
├── code/
├── data/private_mil/
│   ├── images/{train,val,test}/0001...0290/
│   ├── patient_info.csv
│   ├── patient_manifest.csv
│   ├── image_manifest.csv
│   └── split_summary.json
├── slurm/
└── runs/
```

患者文件夹和图像文件均已脱敏重命名；本地原始文件不改变。

## 模型

每张图像经过共享的 ImageNet 预训练 ResNet 主干，随后使用 gated-attention MIL 聚合为患者级特征并输出一个概率。每位患者只计算一次损失。训练最多随机采样16张图像，验证与测试使用该患者全部图像。主干可选 `resnet18` 或 `resnet50`；小样本实验优先使用参数更少的 `resnet18`。

两个训练模型使用相同数据、相同初始化种子和相同超参数，只改变优化器：

- `AdamW`：学习率 `1e-4`
- `SGD + momentum + Nesterov`：学习率 `1e-3`
- 最大训练 `120` 轮；验证集 PR-AUC 未提高至少 `0.001` 达连续 `15` 轮时提前停止

## 服务器验证与提交

```bash
cd /appsnew/home/jlpan/ResNet
/appsnew/home/jlpan/.conda/envs/usfm-thyroid/bin/python \
  code/validate_package.py --data-root data/private_mil --decode-sample 30

bash slurm/submit_two_optimizers.sh
```

ResNet-18 两优化器实验：

```bash
BACKBONE=resnet18 SEED=20260811 WEIGHTS=imagenet \
  bash slurm/submit_two_optimizers.sh
```

也可分别提交：

```bash
sbatch --export=ALL,OPTIMIZER=adamw,SEED=20260811,WEIGHTS=imagenet \
  /appsnew/home/jlpan/ResNet/slurm/train_optimizer.slurm

sbatch --export=ALL,OPTIMIZER=sgd,SEED=20260811,WEIGHTS=imagenet \
  /appsnew/home/jlpan/ResNet/slurm/train_optimizer.slurm
```

监控：

```bash
squeue -j ADAMW_JOB_ID,SGD_JOB_ID -o "%.18i %.20j %.2t %.10M %.6D %R"
sacct -j ADAMW_JOB_ID,SGD_JOB_ID \
  --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES
```

每个模型结果位于 `runs/<optimizer>_<job_id>/`，必须同时存在：

- `best.pt`
- `summary_metrics.json`
- `history.csv`
- `val_predictions.csv`
- `test_predictions.csv`
- `val_attention.csv`
- `test_attention.csv`

模型比较应以验证集 PR-AUC 选择优化器，测试集只作最终一次报告。
