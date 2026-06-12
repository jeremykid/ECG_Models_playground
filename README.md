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

## Architecture Sources

- `Torch_ConvNext.py` is adapted from
  [`comp-well-org/ESI/model/convnextv2.py`](https://github.com/comp-well-org/ESI/blob/main/model/convnextv2.py).
- `Torch_ResNet.py` is adapted from
  [`antonior92/ecg-age-prediction/resnet.py`](https://github.com/antonior92/ecg-age-prediction/blob/main/resnet.py).
- `Torch_PreOpNet.py` is adapted from
  [`ecg-net/PreOpNet/models.py`](https://github.com/ecg-net/PreOpNet/blob/main/models.py).
- `Torch_ECG_CPC.py` is adapted from
  [`AI4HealthUOL/ecg-fm-benchmarking`](https://github.com/AI4HealthUOL/ecg-fm-benchmarking)
  and wraps its ECG-CPC model/checkpoint loading path.
- `Torch_Mamba1D.py` and `Torch_Medformer.py` are adapted from
  [stevenah/deepecg-kit](https://github.com/stevenah/deepecg-kit), specifically:

  - [`deepecgkit/models/mamba1d.py`](https://github.com/stevenah/deepecg-kit/blob/main/deepecgkit/models/mamba1d.py)
  - [`deepecgkit/models/medformer.py`](https://github.com/stevenah/deepecg-kit/blob/main/deepecgkit/models/medformer.py)

See `THIRD_PARTY_NOTICES.md` for source license notes.

## References

- Yu H, Guo P, Sano A. "ECG Semantic Integrator (ESI): A Foundation ECG Model
  Pretrained with LLM-Enhanced Cardiological Text." TMLR, 2024.
  https://openreview.net/forum?id=giEbq8Khcf
- Woo S, Debnath S, Hu R, Chen X, Liu Z, Kweon IS, Xie S. "ConvNeXt V2:
  Co-designing and Scaling ConvNets with Masked Autoencoders." CVPR, 2023.
  https://arxiv.org/abs/2301.00808
- Lima E.M., Ribeiro A.H., Paixao G.M.M. et al. "Deep neural
  network-estimated electrocardiographic age as a mortality predictor." Nature
  Communications, 2021.
  https://www.nature.com/articles/s41467-021-25351-7
- Ouyang D, Theurer J, Stein N.R. et al. "Electrocardiographic Deep Learning
  for Predicting Post-Procedural Mortality." 2022.
  https://arxiv.org/abs/2205.03242
- Al-Masud M.A., Lopez Alcaraz J.M., Strodthoff N. "Benchmarking ECG
  Foundational Models: A Reality Check Across Clinical Tasks." arXiv:2509.25095,
  2025.
  https://arxiv.org/pdf/2509.25095
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
