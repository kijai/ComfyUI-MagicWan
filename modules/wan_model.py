import torch
from einops import repeat

from comfy.ldm.wan.model import WanModel, sinusoidal_embedding_1d


class ModifiedWanModel(WanModel):
    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options={},
    ):
        """
        Modified forward pass to support FlowEdit's dual conditioning
        """
        # Store original shape for unpatchifying
        original_shape = list(x.shape)
        transformer_options['original_shape'] = original_shape
        
        # Embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # Time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x.dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # Context processing
        context = self.text_embedding(context)

        # Handle clip features for I2V if provided
        if clip_fea is not None and self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
            
        # Store context size for attention blocks
        if 'txt_size' not in transformer_options:
            transformer_options['txt_size'] = context.shape[1]
            if clip_fea is not None and self.img_emb is not None:
                transformer_options['txt_size'] = context.shape[1] - 257  # Account for clip tokens

        # Process through attention blocks
        kwargs = dict(
            e=e0,
            freqs=freqs,
            context=context,
            transformer_options=transformer_options)

        for block in self.blocks:
            x = block(x, **kwargs)

        # Final head
        x = self.head(x, e)

        # Unpatchify
        x = self.unpatchify(x, grid_sizes)
        
        # Crop to original dimensions if needed
        if list(x.shape[2:]) != original_shape[2:]:
            x = x[:, :, :original_shape[2], :original_shape[3], :original_shape[4]]
            
        return x

    def forward(self, x, timestep, context, clip_fea=None, transformer_options={}, **kwargs):
        bs, c, t, h, w = x.shape
        
        # Handle padding to patch size if needed
        from comfy.ldm.common_dit import pad_to_patch_size
        x = pad_to_patch_size(x, self.patch_size)
        
        # Calculate positional embeddings
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
        img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

        # Calculate rope frequencies
        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        
        # Regional conditioning support (similar to HunyuanVideo)
        regional_conditioning = transformer_options.get('patches', {}).get('regional_conditioning', None)
        if regional_conditioning is not None:
            context = regional_conditioning[0](context, transformer_options)
            
        return self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs, transformer_options=transformer_options)


def inject_model(diffusion_model):
    """
    Replace the model class with our modified version
    """
    diffusion_model.__class__ = ModifiedWanModel
    return diffusion_model
