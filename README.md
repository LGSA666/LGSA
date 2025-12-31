# LGSA: Label Geometry Structuring and Aligning for Hierarchical Text Classification
This repository implements LGSA, an optimized model via general orthogonal frame and geometric regularization for hierarchical text classification.

## Preprocess
For details about data acquisition, processing, and baseline parameter settings, please refer to HPT.

## Train
Checkpoints are in `./checkpoints/DATA-NAME`. Two checkpoints are kept based on macro-F1 and micro-F1 respectively (`checkpoint_best_macro.pt`, `checkpoint_best_micro.pt`).
The training requires the modification of parameters based on the dataset. `--gof_loss_weight` is for the wight of hierarchical geometric regularization loss, and `--label-loss-wight`, `--hie-label-loss-wight` are for the wight of singular value
smoothing regularization loss.
We take the main results as the average of six random experiments.
### Elamples
```
python train.py --name test --batch 16 --data WebOfScience
```

## Test
Use `--extra _macro` or `--extra _micro` to choose from using `checkpoint_best_macro.pt` or `checkpoint_best_micro.pt` respectively.
### Elamples
```
python test.py --name WebOfScience-test
```
