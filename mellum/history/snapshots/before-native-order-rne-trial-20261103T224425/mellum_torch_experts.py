# SPDX-License-Identifier: MIT
"""Mellum native backbone with an opt-in, unshuffled torch expert backend.

The Qwen MoE container, native routing and checkpoint loaders are retained.
Only expert matrix operations/combination and their post-load layout differ.
The loop follows the existing local naive baseline. Optional route-weight BF16
and model-scoped RNE normalization are controlled independently.
"""
import torch
import torch.nn.functional as F
from aiter import ActivationType

from atom.model_ops.moe import FusedMoE, UnquantizedFusedMoEMethod
from atom.models.mellum import MellumForCausalLM
from atom.models.mellum_rne_norm import install_mellum_rne_norms
from atom.utils import envs


def torch_expert_forward(hidden, w13, w2, indices, weights):
    """Run selected experts in ascending expert order, preserving route dtype."""
    result = torch.zeros_like(hidden)
    mask = F.one_hot(indices.long(), num_classes=w13.shape[0]).permute(2, 1, 0)
    for expert in indices.unique(sorted=True).tolist():
        position, tokens = torch.where(mask[expert])
        gate, up = F.linear(hidden[tokens], w13[expert]).chunk(2, dim=-1)
        states = F.linear(F.silu(gate) * up, w2[expert])
        states = states * weights[tokens, position, None]
        # Each expert has unique token indices; no duplicate-index reduction.
        result.index_add_(0, tokens, states.to(result.dtype))
    return result


class TorchExpertMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe):
        super().__init__(moe)
        self.route_weights_bf16 = envs.ATOM_MELLUM_ROUTE_WEIGHT_BF16

    def process_weights_after_loading(self, layer):
        # The native loader already packed [gate; up]. Ordinary F.linear needs
        # this standard layout, so do not execute the AITER shuffle step.
        if layer.w13_weight.dtype != torch.bfloat16 or layer.w2_weight.dtype != torch.bfloat16:
            raise ValueError('Mellum torch experts currently require BF16 weights')
        self.standard_layout = True

    def apply(self, layer, x, router_logits, top_k, renormalize, **options):
        if options.get('expert_map') is not None or options.get('apply_router_weight_on_input', False):
            raise ValueError('Mellum torch experts require local output-weighted experts')
        if options.get('activation', ActivationType.Silu) != ActivationType.Silu:
            raise ValueError('Mellum torch experts require SiLU')
        weights, indices = FusedMoE.select_experts(
            hidden_states=x, router_logits=router_logits, top_k=top_k,
            renormalize=renormalize, use_grouped_topk=options.get('use_grouped_topk', False),
            topk_group=options.get('topk_group'), num_expert_group=options.get('num_expert_group'),
            custom_routing_function=options.get('custom_routing_function'),
            scoring_func=options.get('scoring_func', 'softmax'),
            e_score_correction_bias=options.get('e_score_correction_bias'),
            num_routing_experts=options.get('global_num_experts', -1),
            num_fused_shared_experts=layer.num_fused_shared_experts,
            fused_shared_experts_scoring_func=options.get('fused_shared_experts_scoring_func'),
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        if self.route_weights_bf16:
            weights = weights.to(torch.bfloat16)
        return torch_expert_forward(x, layer.w13_weight, layer.w2_weight, indices, weights)


class MellumTorchExpertsForCausalLM(MellumForCausalLM):
    def __init__(self, atom_config, prefix=''):
        if (atom_config.tensor_parallel_size != 1 or atom_config.parallel_config.data_parallel_size != 1
                or getattr(atom_config, 'pipeline_parallel_size', 1) != 1
                or atom_config.enable_expert_parallel):
            raise ValueError('Mellum torch experts currently support single-GPU inference')
        if not atom_config.enforce_eager or atom_config.compilation_config.level != 0:
            raise ValueError('Mellum torch experts currently require eager / level 0')
        if atom_config.speculative_config is not None or atom_config.enable_tbo:
            raise ValueError('Mellum torch experts currently require MTP/TBO off')
        super().__init__(atom_config, prefix)
        self.rne_norm_count = (
            install_mellum_rne_norms(self) if envs.ATOM_MELLUM_RNE_NORM else 0
        )
        for layer in self.model.layers:
            experts = getattr(layer.mlp, 'experts', None)
            if experts is None:
                continue
            if type(experts.quant_method) is not UnquantizedFusedMoEMethod or experts.online_quant:
                raise ValueError('Mellum torch experts require unquantized native expert storage')
            experts.quant_method = TorchExpertMethod(experts.moe_config)
