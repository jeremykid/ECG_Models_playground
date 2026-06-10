# ECG_Models_playground

A lightweight PyTorch playground for exploring ECG deep learning model
architectures. Most files are standalone model definitions that can be imported
directly into experiments.

## Model Files

- `Torch_ResNet.py` - 1D ResNet variants, including feature-fusion support.
- `Torch_ViT1D.py` - 1D Vision Transformer variants.
- `Torch_ConvNext.py` - ConvNeXt V2 adapted to 1D ECG signals.
- `Torch_CNN_Transformer.py` - CNN front end with Transformer sequence blocks.
- `Torch_PreOpNet.py` - PreOpNet / EffNet-style ECG encoder and classifier.
- `Torch_XResNet1D.py` - XResNet-style 1D backbone.
- `basic_conv1d.py` - FCN, SEN, and basic 1D convolutional helpers.
- `Torch_ResNet_Multi_Attention.py` - ResNet with multi-head attention.
- `Torch_ECG_CPC.py` - ECG-CPC wrapper that depends on external checkpoints.
- `Torch_Mamba1D.py` - Bidirectional Mamba-style ECG classifier.
- `Torch_Medformer.py` - Multi-granularity patching Transformer classifier.
- `Hubert_ECG/` - HuBERT-ECG MIMIC-IV dataset and fine-tuning scripts.

## New Architectures From DeepECG-Kit

`Torch_Mamba1D.py` and `Torch_Medformer.py` are adapted from
[stevenah/deepecg-kit](https://github.com/stevenah/deepecg-kit), specifically:

- [`deepecgkit/models/mamba1d.py`](https://github.com/stevenah/deepecg-kit/blob/main/deepecgkit/models/mamba1d.py)
- [`deepecgkit/models/medformer.py`](https://github.com/stevenah/deepecg-kit/blob/main/deepecgkit/models/medformer.py)

DeepECG-Kit is licensed under Apache-2.0.

## References

- Gu A, Dao T. "Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces." arXiv:2312.00752, 2023.
  https://arxiv.org/abs/2312.00752
- Wang N, Liang X, Wang Z, Zhao J, Liu Y, Peng L, Miao C. "Medformer: A
  Multi-Granularity Patching Transformer for Medical Time-Series
  Classification." NeurIPS, 2024.
  https://papers.neurips.cc/paper_files/paper/2024/hash/3fe2a777282299ecb4f9e7ebb531f0ab-Abstract-Conference.html

## Smoke Test

```python
import torch

from Torch_Mamba1D import Mamba1D
from Torch_Medformer import Medformer

x = torch.randn(2, 12, 500)

mamba = Mamba1D(input_channels=12, output_size=5, d_model=32, num_layers=1)
medformer = Medformer(input_channels=12, output_size=5, d_model=32, num_encoder_layers=1, nhead=4)

print(mamba(x).shape)
print(medformer(x).shape)
```
