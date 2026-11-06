# SPDX-License-Identifier: MIT
"""Mellum-scoped RNE RMSNorm, including the BF16 residual boundary.

The existing native RMSNorm parameters/loaders are retained. This correctness
path uses ordinary torch operations under the single-GPU eager Mellum adapter;
it neither changes AITER nor the shared ATOM norm implementation.
"""
import torch

from atom.model_ops.layernorm import RMSNorm


class MellumRNERMSNorm(RMSNorm):
    def __init__(self, dim, eps=1e-6, *, hf_rounding=False):
        super().__init__(dim, eps=eps)
        self.hf_rounding = hf_rounding

    def forward(self, x, residual=None, x_scale=None):
        if x_scale is not None or self.use_fused_quant or self.x_pad_to_multiple:
            raise ValueError("Mellum RNE norm requires unquantized, unpadded inputs")
        if self.fused_allreduce and self.tp_size > 1:
            raise ValueError("Mellum RNE norm currently requires a single GPU")
        input_dtype = x.dtype
        if residual is None:
            values = x.float()
        elif self.hf_rounding:
            # Reference mode: round the residual before normalization.
            residual_out = x + residual
            values = residual_out.float()
        else:
            # Native order: normalize the FP32 sum, store an RNE BF16 residual.
            values = x.float() + residual.float()
            residual_out = values.to(input_dtype)
        variance = values.pow(2).mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps)
        if self.hf_rounding:
            output = self.weight * normalized.to(input_dtype)
        else:
            # Retain native FP32 scaling; change only the output conversion.
            output = (normalized * self.weight.float()).to(input_dtype)
        return output if residual is None else (output, residual_out)


def install_mellum_rne_norms(model, *, hf_rounding=False):
    """Replace native norm modules while preserving each original Parameter."""
    count = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if type(child) is not RMSNorm:
                continue
            if child.use_fused_quant or child.x_pad_to_multiple or child.tp_size != 1:
                raise ValueError("Unsupported native norm configuration for Mellum RNE")
            replacement = MellumRNERMSNorm(child.dim, eps=child.eps, hf_rounding=hf_rounding)
            replacement.weight = child.weight
            assert replacement.weight is child.weight
            setattr(module, name, replacement)
            count += 1
    return count
