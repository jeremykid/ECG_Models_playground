"""
ECG-CPC (Contrastive Predictive Coding) Model
Extracted and adapted from the ecg-fm-benchmarking repository:
https://github.com/AI4HealthUOL/ecg-fm-benchmarking

Paper: Al-Masud, Lopez Alcaraz, and Strodthoff, "Benchmarking ECG
Foundational Models: A Reality Check Across Clinical Tasks", arXiv:2509.25095

This is a simplified standalone version for integration with the main ECG multi-task framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys
from dataclasses import dataclass

# Add ecg-fm-benchmarking to path
ECG_FM_BENCH_PATH = Path(__file__).parent.parent.parent / "ecg-fm-benchmarking" / "code"
if str(ECG_FM_BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(ECG_FM_BENCH_PATH))

try:
    from clinical_ts.models.ecg_foundation_models.ecg_cpc.basic_io import load_model_from_config
    from clinical_ts.template_model import BaseModel
    CPC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ECG-CPC dependencies not available: {e}")
    print("Please ensure ecg-fm-benchmarking is properly set up.")
    CPC_AVAILABLE = False


class ECGCPCEncoder(nn.Module):
    """
    Simplified ECG-CPC Encoder for feature extraction.
    
    Architecture:
    - 4-layer convolutional encoder (RNN-style naming but actually CNNs)
    - S4 (Structured State Space) predictor module
    - Outputs 512-dimensional embeddings
    
    Args:
        input_channels: Number of ECG leads (default: 12)
        num_classes: Output embedding dimension (default: 512)
        seq_length: Input sequence length (default: 2400 for 240Hz, 10s)
        num_features: Number of additional features (age/sex) (default: 0)
        pretrained_config_path: Path to pretrained CPC config yaml
    """
    
    def __init__(
        self,
        input_channels=12,
        num_classes=512,
        seq_length=2400,
        num_features=0,
        pretrained_config_path=None,
        **kwargs
    ):
        super().__init__()
        
        if not CPC_AVAILABLE:
            raise ImportError(
                "ECG-CPC dependencies not available. "
                "Please ensure ecg-fm-benchmarking is properly installed."
            )
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.num_features = num_features
        self.feature_dim = 512  # CPC outputs 512-d features
        self.use_pretrained = bool(kwargs.pop("use_pretrained", True))
        
        # Load the CPC model from config
        if pretrained_config_path is None:
            # Use default config path
            default_config = Path(__file__).parent.parent.parent / \
                           "ecg-fm-benchmarking" / "checkpoints" / "ECG-CPC Checkpoint" / \
                           "config_last_11597276_ckpt.yaml"
            if not default_config.exists():
                raise FileNotFoundError(
                    f"CPC config not found at {default_config}. "
                    f"Please ensure the checkpoint folder is in: "
                    f"ecg-fm-benchmarking/checkpoints/ECG-CPC Checkpoint/"
                )
            pretrained_config_path = str(default_config)

        runtime_overrides = self._prepare_runtime_overrides(
            Path(pretrained_config_path),
            use_pretrained=self.use_pretrained,
        )

        print(f"Loading ECG-CPC model from: {pretrained_config_path}")
        self.cpc_model, self.cpc_config = load_model_from_config(
            config_name=str(pretrained_config_path),
            overrides=runtime_overrides,
        )
        
        # Freeze CPC encoder by default (for feature extraction)
        for param in self.cpc_model.parameters():
            param.requires_grad = False
        
        # Optional: Add projection head for different output dimensions
        if num_classes != self.feature_dim:
            self.projection = nn.Linear(self.feature_dim, num_classes)
        else:
            self.projection = nn.Identity()
        
        # Optional: Add age/sex features
        if num_features > 0:
            self.dense_agsx = nn.Linear(num_features, 10)
            self.final_projection = nn.Linear(num_classes + 10, num_classes)
        else:
            self.dense_agsx = None
            self.final_projection = None
    
    def forward(self, input_tensor):
        """
        Forward pass through ECG-CPC encoder.
        
        Args:
            input_tensor: If num_features > 0, tuple of (ecg, age_sex)
                         Otherwise, just ecg tensor
                         ecg shape: (batch, leads, seq_length)
        
        Returns:
            embeddings: (batch, num_classes)
        """
        if self.num_features > 0:
            x, age_sex = input_tensor[0], input_tensor[1]
        else:
            x = input_tensor
        
        # Ensure input is the right shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Handle NaN values
        x = torch.nan_to_num(x)
        
        # CPC model expects specific input format
        # The model processes: (batch, channels, time)
        batch_size = x.shape[0]
        
        # Forward through CPC encoder
        # CPC returns dict with 'seq' key containing sequence features
        output = self.cpc_model(seq=x)
        sequence_features = output["seq"]  # (batch, seq_len, feature_dim)
        
        # Global average pooling over sequence dimension
        pooled_features = sequence_features.mean(dim=1)  # (batch, feature_dim)
        
        # Project to desired output dimension
        embeddings = self.projection(pooled_features)
        
        # Optionally incorporate age/sex features
        if self.num_features > 0 and self.dense_agsx is not None:
            age_sex_emb = self.dense_agsx(age_sex)
            combined = torch.cat([embeddings, age_sex_emb], dim=1)
            embeddings = self.final_projection(combined)
        
        return embeddings
    
    def unfreeze_encoder(self):
        """Unfreeze the CPC encoder for fine-tuning."""
        for param in self.cpc_model.parameters():
            param.requires_grad = True
    
    def freeze_encoder(self):
        """Freeze the CPC encoder (for feature extraction only)."""
        for param in self.cpc_model.parameters():
            param.requires_grad = False

    @staticmethod
    def _ensure_stub_ptbxl_metadata(stub_dir: Path) -> Path:
        """Create a tiny PTB-XL metadata stub so checkpoint loading does not depend on local benchmark data."""
        stub_dir.mkdir(parents=True, exist_ok=True)
        df_path = stub_dir / "df_memmap.pkl"
        lbl_path = stub_dir / "lbl_itos.pkl"

        if not df_path.exists():
            dummy_df = pd.DataFrame(
                {
                    "strat_fold": ["x-1"],
                    "label_all": [["DUMMY_LABEL"]],
                    "label_all_filtered_numeric": [[0]],
                }
            )
            dummy_df.to_pickle(df_path)

        if not lbl_path.exists():
            dummy_labels = np.array(["DUMMY_LABEL"], dtype=object)
            dummy_lbl_itos = {
                "label_all": dummy_labels,
                "label_diag": dummy_labels,
                "label_form": dummy_labels,
                "label_rhythm": dummy_labels,
                "label_diag_subclass": dummy_labels,
                "label_diag_superclass": dummy_labels,
            }
            with open(lbl_path, "wb") as handle:
                pickle.dump(dummy_lbl_itos, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return stub_dir

    @classmethod
    def _prepare_runtime_overrides(cls, config_path: Path, use_pretrained: bool) -> list[str]:
        """Build Hydra overrides that replace placeholder paths from the published checkpoint config."""
        if not config_path.exists():
            raise FileNotFoundError(f"CPC config not found: {config_path}")

        checkpoint_path = config_path.parent / "last_11597276.ckpt"
        checkpoint_alias_dir = config_path.parent.parent / "ECG-CPC-Checkpoint"
        if not checkpoint_alias_dir.exists():
            try:
                checkpoint_alias_dir.symlink_to(config_path.parent, target_is_directory=True)
            except OSError:
                checkpoint_alias_dir = config_path.parent
        alias_checkpoint_path = checkpoint_alias_dir / checkpoint_path.name

        overrides = []
        if use_pretrained:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"CPC checkpoint not found: {checkpoint_path}")
            overrides.append(f"trainer.pretrained='{alias_checkpoint_path.as_posix()}'")
        else:
            overrides.append("trainer.pretrained=''")

        stub_dir = cls._ensure_stub_ptbxl_metadata(config_path.parent.parent / "ecg_cpc_ptbxl_stub")
        overrides.append(f"data0.path='{stub_dir.as_posix()}'")
        return overrides
