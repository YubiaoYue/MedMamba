# MedMamba 本地化分支（Windows / 路线 A）

本仓库是 [YubiaoYue/MedMamba](https://github.com/YubiaoYue/MedMamba) 的本地化分支，用于智能医学图像处理实验课的医学图像分类实验。原仓库 README 与 requirements.txt 已删除，改为记录本机的实际适配方案。

## 上游与同步基线

| 项 | 值 |
|---|---|
| 上游仓库 | `YubiaoYue/MedMamba`（remote 名 `upstream`，push 已锁为 `NO_PUSH`，只读） |
| 同步基线 commit | `104beb3d641d67cef6f09b94ac299083d91bc417` |
| 本仓库（fork） | `Yui-0818/MedMamba_fole`（remote 名 `origin`） |
| 工作分支 | `routeA-windows` |

`git diff main` 与 `git show main:MedMamba.py` 可随时取回未改动的原版做对照。

## 运行环境

- conda 环境 `IMIP`，Python 3.10.22
- GPU：NVIDIA RTX 4060 Laptop（8GB，sm_89），CUDA runtime 12.4，cuDNN 9.10
- 系统：Windows，**无** CUDA Toolkit / MSVC / WSL

安装分两步（顺序不能合并，`--index-url` 会屏蔽 PyPI 上的其他依赖）：

```
python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## 与上游的关键差异：路线 A（零编译）

上游 `MedMamba.py` 第 12—21 行把 `mamba_ssm` 与 `selective_scan` 的导入包在 `try/except: pass` 中，缺失时不在安装期报错，而是在首次前向的 `forward_corev0` 里以 `NameError: selective_scan_fn is not defined` 暴露。`mamba_ssm`、`causal_conv1d` 在 PyPI 上只提供源码包，Windows 需 nvcc + MSVC 编译，本机不具备。

因此本分支不使用 CUDA 融合算子，改以**纯 PyTorch 实现的 `selective_scan_ref`** 顶替 `selective_scan_fn`。两者函数签名一致，对模型结构零侵入，参数量与四方向扫描（K=4, d_state=16）不变。

代价：参考实现沿序列长度 L 逐步循环并显式物化 K×d×n×L 的 fp32 中间状态，224×224 输入下第一 stage 单样本状态张量约 77MB，故训练吞吐显著低于 CUDA 算子，且需下调 batch size；该路线适用于小规模数据的验证与演示，不用于复现论文 100 epoch 的全量训练。

上游 requirements.txt 另有一处错漏：未列出 `MedMamba.py` 第 10 行实际需要的 `einops`，本仓库的 requirements.txt 已补入。

## 数据集

本实验数据不入库，位于仓库外的 `Experiment1/data/`：

- 三分类，标签写在 `train.csv` / `test.csv`（表头 `image,lable`）
- 训练集 569 张（类 0/1/2 = 286/210/73），测试集 143 张（72/53/18），类别不均衡
- 原图为 RGB 2592×1728，需降采样并按类别重排为 `ImageFolder` 目录结构后使用

## 当前进度

- [x] conda 环境 `IMIP` 与依赖安装，`pip check` 无冲突
- [x] 从 fork 克隆、挂载 upstream、建立工作分支 `routeA-windows`
- [x] 删除上游 README.md / requirements.txt，替换为本地版本
- [ ] `selective_scan_ref` 纯 PyTorch 实现替换
- [ ] `MedMamba.py` 末尾 import-time 建模型并 `.to("cuda")` 的演示段隔离
- [ ] 数据按类别重排
- [ ] `train_imip.py`（上游 `train.py` 含语法错误与未定义变量，不直接改它）
- [ ] 前向反向 smoke test 与训练指标

## 引用

论文：*MedMamba: Vision Mamba for Medical Image Classification*, arXiv:2403.03849。模型结构与算法贡献归上游作者所有。

```bibtex
@article{yue2024medmamba,
  title={MedMamba: Vision Mamba for Medical Image Classification},
  author={Yue, Yubiao and Li, Zhenzhang},
  journal={arXiv preprint arXiv:2403.03849},
  year={2024}
}
```
