from ..modules.wan_model import inject_model
from ..modules.wan_blocks import inject_blocks


class ConfigureModifiedWanNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
        }}
    RETURN_TYPES = ("MODEL",)

    CATEGORY = "magicwan"
    FUNCTION = "apply"

    def apply(self, model):
        # Inject modified model and block classes
        inject_model(model.model.diffusion_model)
        inject_blocks(model.model.diffusion_model)
        return (model,)
