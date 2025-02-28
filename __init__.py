from .nodes.wan_flowedit_nodes import WanFlowEditGuiderNode, WanFlowEditGuiderAdvNode, WanFlowEditGuiderCFGNode, WanFlowEditGuiderCFGAdvNode, WanFlowEditSamplerNode
from .nodes.modify_wan_model_node import ConfigureModifiedWanNode
from .nodes.wan_model_pred_nodes import WanInverseModelSamplingPredNode, WanReverseModelSamplingPredNode
from .nodes.wan_feta_enhance_node import WanFetaEnhanceNode

NODE_CLASS_MAPPINGS = {
    "WanFlowEditGuider": WanFlowEditGuiderNode,
    "WanFlowEditGuiderAdv": WanFlowEditGuiderAdvNode,
    "WanFlowEditGuiderCFG": WanFlowEditGuiderCFGNode,
    "WanFlowEditGuiderCFGAdv": WanFlowEditGuiderCFGAdvNode,
    "WanFlowEditSampler": WanFlowEditSamplerNode,
    "ConfigureModifiedWan": ConfigureModifiedWanNode,
    "WanInverseModelSamplingPred": WanInverseModelSamplingPredNode,
    "WanReverseModelSamplingPred": WanReverseModelSamplingPredNode,
    "WanFetaEnhance": WanFetaEnhanceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanFlowEditGuider": "Wan FlowEdit Guider",
    "WanFlowEditGuiderAdv": "Wan FlowEdit Guider Advanced",
    "WanFlowEditGuiderCFG": "Wan FlowEdit Guider CFG",
    "WanFlowEditGuiderCFGAdv": "Wan FlowEdit Guider CFG Advanced",
    "WanFlowEditSampler": "Wan FlowEdit Sampler",
    "ConfigureModifiedWan": "Configure Modified Wan Model",
    "WanInverseModelSamplingPred": "Wan Inverse Model Pred",
    "WanReverseModelSamplingPred": "Wan Reverse Model Pred",
    "WanFetaEnhance": "Wan Feta Enhance",
}