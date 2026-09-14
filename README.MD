# ResNet-18 患者级多图像 MIL：训练代码与可复现演示

本项目将原有训练包纳入 Git 版本管理，使用 `results.ipynb` 展示可实际运行的结果。
核心模型是 ResNet 图像编码器与 gated-attention 多实例学习（MIL）聚合器。
仓库包含原始训练脚本、CPU 演示程序、Jupyter Notebook、固定版本依赖和 Git 标签。

## 结果的含义

- `results.ipynb` 使用固定种子生成的合成图像，并实际训练注意力层与分类头。
- 演示冻结随机初始化的 ResNet-18 编码器，不下载预训练权重，在 CPU 上运行。
- 演示图表、概率、注意力和指标都来自本次程序执行；不是私有数据训练结果。
- 合成任务十分简单，其指标不能用于评价医学模型，也不能与真实实验比较。
- 原始训练包未提供训练后的结果文件，因此本仓库没有声称已复现真实训练指标。

## 项目结构

```text
.
├── README.MD                     # 项目说明和复现步骤
├── results.ipynb                 # 已执行、含输出的 Jupyter 展示
├── requirements.txt              # CPU 演示的固定版本依赖
├── requirements-lock.txt         # 已验证环境的完整依赖版本
├── code/
│   ├── train_resnet_mil.py        # 原始训练实现，支持 ResNet-18/50
│   ├── reproduce_demo.py          # 离线合成数据 CPU 演示
│   ├── compare_models.py          # 原始两优化器结果比较
│   └── validate_package.py        # 原始私有数据完整性验证
├── slurm/                        # 原有服务器调度脚本
├── README_服务器训练.md            # 保留的原始服务器说明
├── .gitattributes                # 跨平台文本换行规则
└── .gitignore                    # 排除缓存、权重、私有数据与运行输出
```

## 复现 Jupyter 展示（Python 3.11）

先克隆仓库，进入项目根目录，然后在 Windows PowerShell 运行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c requirements-lock.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name resnet18-demo --display-name "ResNet18 demo"
.\.venv\Scripts\python.exe -m jupyterlab results.ipynb
```

在 Jupyter 中选择 `ResNet18 demo` 内核，使用 Restart Kernel and Run All。
首次安装依赖需要网络；安装完成后，Notebook 本身不需要网络、GPU 或私有数据。
Linux 对应解释器路径是 `.venv/bin/python`，虚拟环境命令为 `python3.11 -m venv .venv`。
锁文件记录 Windows/Python 3.11 验证环境；其他平台先使用 `requirements.txt` 安装。

也可以直接运行演示：

```powershell
.\.venv\Scripts\python.exe code/reproduce_demo.py --output-dir outputs/demo
```

生成的 `outputs/demo/demo_results.json` 包含损失、验证集阈值和测试指标。
Notebook 还展示损失曲线、测试概率、混淆矩阵和两幅图像的注意力权重。
`outputs/` 不纳入 Git；Notebook 保留已执行输出，便于在仓库网页直接查看。

## 可复现性设置

- 固定种子为 `20260811`，训练、验证、测试合成图像使用三个不同的种子。
- 使用 CPU、两个计算线程和 PyTorch 确定性算法。
- 演示固定训练 6 轮；只使用验证集确定分类阈值，测试集用于最后报告。
- `pr_auc` 沿用原始代码字段，实际计算的是 Average Precision（AP）。
- 同一验证环境连续运行应得到完全相同的输出；跨硬件或库版本可能有浮点差异。
- 真实训练入口默认骨干为 ResNet-50，因此复现 ResNet-18 时必须显式传入 `--backbone resnet18`。

## 真实数据训练（需另行准备私有数据和 CUDA 环境）

原始训练要求 290 位患者、3,706 张图像；划分固定为训练 174、验证 58、测试 58。
这些数据未包含在 Git 仓库中。只有获得原始数据及相同清单，才能复现真实实验。
准备 `data/private_mil/patient_manifest.csv`、`image_manifest.csv` 和清单对应图像。
原始数据字段和服务器流程见 `README_服务器训练.md`。

在单独的 CUDA 环境中使用原始脚本；不要使用上面的 CPU 依赖环境执行完整训练。
需要兼容的 PyTorch 2.5.1 / torchvision 0.20.1 CUDA 构建及其余计算依赖。
服务器脚本包含原机器路径和分区名称，使用前按自己的环境修改。

```bash
python code/validate_package.py --data-root data/private_mil --decode-sample 30
python code/train_resnet_mil.py --data-root data/private_mil --output-dir runs/resnet18_adamw --optimizer adamw --backbone resnet18 --seed 20260811 --weights imagenet
python code/train_resnet_mil.py --data-root data/private_mil --output-dir runs/resnet18_sgd --optimizer sgd --backbone resnet18 --seed 20260811 --weights imagenet
python code/compare_models.py --adamw runs/resnet18_adamw --sgd runs/resnet18_sgd --output outputs/comparison.json
```

真实训练使用 ImageNet 权重，需要网络下载或已有的 `.cache/torch` 缓存。
若使用项目内缓存，请设置 `TORCH_HOME` 为项目的 `.cache/torch` 绝对路径。
原目录已有 `resnet18-f37072fd.pth`，SHA-256 为
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。
该缓存保留在本地，被 Git 忽略，不随仓库上传。
完整训练应保存环境、输入清单校验和、运行配置及原始结果，选择优化器时仅依据验证集。

## Git 版本管理与作业提交

首次提交保存原始源码和忽略规则，第二次提交增加可复现演示与文档。
发行标签为 `v1.0.0`。查看版本与回到提交版本：

```bash
git status
git log --oneline --decorate
git tag -n
git switch --detach v1.0.0
```

日常修改后使用 `git add <文件名>` 和 `git commit -m "说明修改内容"`。
创建自己的 GitHub/Gitee 空仓库后，设置实际远程地址并上传：

```bash
git remote add origin <你的远程仓库地址>
git push -u origin main
git push origin v1.0.0
```

提交作业时提供远程仓库主页链接，并注明 Notebook 为 `results.ipynb`、标签为 `v1.0.0`。
本地路径不能替代老师能访问的远程仓库链接。
