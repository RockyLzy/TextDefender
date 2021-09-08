from torch import nn as nn

class EmbeddingHook:
    forward_value = None
    backward_gradient = None

    @classmethod
    def fw_hook_layers(cls, module, inputs, outputs):
        cls.forward_value = outputs

    @classmethod
    def bw_hook_layers(cls, module, grad_in, grad_out):
        cls.backward_gradient = grad_out[0]

    @classmethod
    def register_embedding_hook(cls, embedding: nn.Embedding):
        fw_hook = embedding.register_forward_hook(cls.fw_hook_layers)
        bw_hook = embedding.register_backward_hook(cls.bw_hook_layers)
        return [fw_hook, bw_hook]

    @classmethod
    def reading_embedding_hook(cls):
        return cls.forward_value, cls.backward_gradient
