import torch
from torch import nn

from comfy.ldm.flux.math import apply_rope
from comfy.ldm.wan.model import WanAttentionBlock, WanT2VCrossAttention, WanI2VCrossAttention


class ModifiedWanAttentionBlock(WanAttentionBlock):
    """
    Modified Wan Attention Block that supports FlowEdit operations
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx = 0  # Will be set by inject_blocks
        
    def forward(
        self,
        x,
        e,
        freqs,
        context,
        transformer_options={},
    ):
        # Get original forward implementation
        original_output = super().forward(x, e, freqs, context)
        
        # If we're in FlowEdit mode and need special handling
        latent_type = transformer_options.get('transformer_options', {}).get('latent_type', None)
        
        # Apply any custom modifications for FlowEdit if needed
        if latent_type is not None:
            # Could add specialized handling here if needed
            pass
            
        return original_output


class ModifiedWanT2VCrossAttention(WanT2VCrossAttention):
    """
    Modified T2V cross attention for FlowEdit support
    """
    def forward(self, x, context, transformer_options={}):
        # Get original implementation
        return super().forward(x, context)


class ModifiedWanI2VCrossAttention(WanI2VCrossAttention):
    """
    Modified I2V cross attention for FlowEdit support
    """
    def forward(self, x, context, transformer_options={}):
        # Get original implementation
        return super().forward(x, context)


def inject_blocks(diffusion_model):
    """
    Replace all attention blocks with our modified versions
    """
    # Replace the attention blocks
    for i, block in enumerate(diffusion_model.blocks):
        block.__class__ = ModifiedWanAttentionBlock
        block.idx = i
        
        # Replace the cross attention mechanisms
        if hasattr(block, 'cross_attn'):
            if block.cross_attn.__class__ == WanT2VCrossAttention:
                block.cross_attn.__class__ = ModifiedWanT2VCrossAttention
            elif block.cross_attn.__class__ == WanI2VCrossAttention:
                block.cross_attn.__class__ = ModifiedWanI2VCrossAttention
                
    return diffusion_model