
class WanFetaEnhanceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "feta_weight": ("FLOAT", {"default": 1, "min": -100.0, "max": 100.0, "step":0.01}),
        }
    }

    RETURN_TYPES = ("MODEL",)

    CATEGORY = "hunyuanloom"
    FUNCTION = "apply"

    def apply(self, model, feta_weight):
        model = model.clone()

        model_options = model.model_options.copy()
        transformer_options = model_options['transformer_options'].copy()

        transformer_options['feta_weight'] = feta_weight
        model_options['transformer_options'] = transformer_options

        model.model_options = model_options
        return (model,)

