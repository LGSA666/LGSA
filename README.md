# SIHTC: Hierarchical Text Classification Optimization via Structural Entropy and Singular Smoothing
This repository implements SIHTC, an optimized model via structural entropy and singular smoothing for hierarchical text classification.

## Preprocess
For details about data acquisition, processing, and baseline parameter settings, please refer to [HPT](https://github.com/wzh9969/HPT).

## Train
Checkpoints are in `./checkpoints/DATA-NAME`. Two checkpoints are kept based on macro-F1 and micro-F1 respectively (`checkpoint_best_macro.pt`, `checkpoint_best_micro.pt`).
The training requires the modification of parameters based on the dataset. `--seloss-wight` is for the wight of structural entropy loss, and `--label-loss-wight`, `--hie-label-loss-wight` are for the wight of singular value
smoothing regularization loss.
We take the main results as the average of six random experiments.
### Elamples
```
python train.py --name test --batch 30 --data WebOfScience --seloss-wight 0.05 --label-loss-wight 0.05 --hie-label-loss-wight 0.05
python train.py --name test --batch 30 --data rcv1 --seloss-wight 0.1 --label-loss-wight 0.005 --hie-label-loss-wight 0.005
python train.py --name test --batch 30 --data NYT --seloss-wight 0.01 --label-loss-wight 0.005 --hie-label-loss-wight 0.005

python train.py --name test --batch 30 --data WebOfScience --label-loss-wight 0.05 --hie-label-loss-wight 0.05
```

## Test
Use `--extra _macro` or `--extra _micro` to choose from using `checkpoint_best_macro.pt` or `checkpoint_best_micro.pt` respectively.
### Elamples
```
python test.py --name WebOfScience-test
```

## Improments
 - 引入超曲空间标签嵌入（Poincaré / Lorentz），让层次结构天然保序，再把你的 GOF 角度/长度约束改写到曲空间；比较曲/欧几里得的增益与开销。
 - 把几何正则做“课程”调度：前期弱约束保证收敛，后期强约束收紧；或按节点深度/置信度自适应权重，而非单一 alpha warmup。
 - 增加对比式监督：同父/兄弟视为正样本，跨子树视为负样本，构造层次感知的 InfoNCE，补充目前的 GOF 目标。
 - 让 ideal_label_embeddings 可选 LoRA/Adapter 微调模式：保持主干冻结但给理想向量加小型可训练低秩偏移，兼顾稳定性与适配    数据分布。
 - 融合描述/名称的多视角标签表示：名称平均、描述编码、图结构信息做门控融合，而不是二选一开关，可学习权重决定依赖程度。
 - 评估与鲁棒性增强：增加层次校准（温度缩放）和不确定性度量；在噪声标签下测试，用一致性正则或软标签平滑缓解噪声。
 - 推理侧优化：父节点剪枝（只扩展高于阈值的子树）或 beam over depth，减少无效计算，同时可测试对精度的影响。
