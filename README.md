# ORCA Redox Potential Calculation Pipeline

基于 ORCA 6.X 的自动化单分子氧化还原电位高通量计算与智能自救分析工作流。

## 🌟 核心特性

- **完整热力学循环**：支持 3 态（中性态 GN、氧化态 OX、还原态 RD）$\times$ 4 步（溶剂化优化+频率、高精度气相单点、基准气相单点、基准液相单点）全自动计算。
- **4-Job Slurm 依赖协同**：
  - `Job 1 (GN)` 优先运行并生成优质基态几何构型。
  - `Job 2 (OX)` 与 `Job 3 (RD)` 并发执行，自动继承 `GN` 的优化坐标大幅加速收敛。
  - **降级回退保证**：若 `GN` 失败，`OX` 和 `RD` 自动回退至初始力场坐标继续计算，避免任务浪费。
  - `Job 4 (Analysis)` 轻量化汇总，即使部分状态失败也会最大化输出已成功的氧化/还原电位报表。
- **智能诊断与自救引擎 (`smart_optimizer.py`)**：
  - **隔离重试**：每次出错自动在 `retry_XX_<reason>` 独立子目录中排查，不破坏原始计算现场。
  - **虚频微扰**：对 $\nu < -50\text{ cm}^{-1}$ 的严重虚频自动提取法向振动位移模态，施加微扰消除过渡态鞍点（忽略 $\ge -50\text{ cm}^{-1}$ 的数值软模）。
  - **优化超限恢复**：从轨迹 `trj.xyz` 中自动提取末帧几何构型并调整步长继续优化。
  - **SCF 梯次自救**：`SlowConv` $\to$ `SOSCF` $\to$ `PModel / Level Shift` 自动切换。

## 🚀 快速上手

### 1. 准备环境
```bash
conda create -n opi python=3.11 -y
conda activate opi
pip install orca-pi pyyaml pandas
```

### 2. 从 SMILES 启动计算
```bash
python orca_redox.py -i "CC(=O)O" -n acetic_acid --cores 32
```

### 3. 从已有 XYZ 坐标启动
```bash
python orca_redox.py -i 1996-88-9.xyz -n 1996-88-9 --cores 32
```

### 4. 仅分析已有结果
```bash
python orca_redox.py -n 1996-88-9 --analysis
```
自动生成 `redox_summary.csv` 与 `redox_summary.json`。

## ⚙️ 配置文件说明 (`config.yaml`)

可自由定制泛函、基组、CPCM 溶剂介电常数及超算 HPC 环境参数。
