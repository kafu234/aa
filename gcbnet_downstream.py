"""LibEER GCBNet adapter for the local SEED/SEED-IV DE evaluation loops."""

import importlib.util
import sys
from pathlib import Path

import torch


_LIBEER_ROOT = Path("/root/LibEER/LibEER")
_LIBEER_MODEL = _LIBEER_ROOT / "models" / "GCBNet.py"
_LIBEER_CONFIG = _LIBEER_ROOT / "config" / "model_param" / "GCBNet.yaml"


def _load_libeer_gcbnet():
    if not _LIBEER_MODEL.is_file():
        raise ImportError(f"LibEER GCBNet implementation not found: {_LIBEER_MODEL}")
    if not _LIBEER_CONFIG.is_file():
        raise ImportError(f"LibEER GCBNet configuration not found: {_LIBEER_CONFIG}")

    root = str(_LIBEER_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    spec = importlib.util.spec_from_file_location(
        "libeer_gcbnet_downstream", _LIBEER_MODEL)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load LibEER GCBNet from {_LIBEER_MODEL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # LibEER stores this as a working-directory-relative path. Make it absolute
    # so the exact LibEER configuration is used when evaluation starts in FMTS.
    module.param_path = str(_LIBEER_CONFIG)
    return module


_gcbnet = _load_libeer_gcbnet()


class GCBNetClassifier(_gcbnet.GCBNet):
    """GCBNet with LibEER's classifier-weight regularizer exposed locally."""

    def __init__(
        self,
        num_electrodes=62,
        in_channels=5,
        num_classes=3,
        l2_lambda=0.001,
    ):
        super().__init__(
            num_electrodes=num_electrodes,
            in_channels=in_channels,
            num_classes=num_classes,
            lamb=l2_lambda,
        )
        self.l2_lambda = l2_lambda

    def regularization_loss(self):
        # Matches SparseL2Regularization(0.001)(model.original_fc.weight)
        # in LibEER/GCBNet_train.py.
        return self.l2_lambda * torch.norm(self.original_fc.weight, p=2)
