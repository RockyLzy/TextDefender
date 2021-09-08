"""
HuggingFace Model Wrapper
--------------------------
"""
import os
import torch
import transformers

import textattack
import numpy as np

from .pytorch_model_wrapper import PyTorchModelWrapper

from typing import List, Tuple
from sklearn.preprocessing import normalize
from args import ProgramArgs
from torch import nn as nn
from transformers import PreTrainedTokenizer, AutoModelForMaskedLM, RobertaTokenizer
from utils.safer import WordSubstitude

class HuggingFaceModelSaferEnsembleWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""
    def __init__(self, 
                args: ProgramArgs, 
                model: nn.Module, 
                tokenizer: PreTrainedTokenizer, 
                batch_size: int = 300):
        self.model = model.to(textattack.shared.utils.device)

        if isinstance(tokenizer, transformers.PreTrainedTokenizer):
            tokenizer = textattack.models.tokenizers.AutoTokenizer(tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.ensemble_method = args.ensemble_method
        self.ensemble = args.predict_ensemble
        self.augmentor = WordSubstitude(args.safer_perturbation_set)

    def _model_predict(self, inputs):
        """Turn a list of dicts into a dict of lists.

        Then make lists (values of dict) into tensors.
        """
        model_device = next(self.model.parameters()).device
        input_dict = {k: [_dict[k] for _dict in inputs] for k in inputs[0]}
        input_dict = {
            k: torch.tensor(v).to(model_device) for k, v in input_dict.items()
        }
        outputs = self.model(**input_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs[0]


    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        randomized_text_input_list = []
        if isinstance(text_input_list[0], tuple) and len(text_input_list[0]) == 2:
            for text_input in text_input_list:
                randomized_text_input = [(text_input[0], sentence) for sentence in self.augmentor.get_perturbed_batch(text_input[1].lower(), rep=self.ensemble)]
                randomized_text_input_list.extend(randomized_text_input)
        else:
            for text_input in text_input_list:
                randomized_text_input_list.extend(self.augmentor.get_perturbed_batch(text_input.lower(),rep=self.ensemble))

        ids = self.encode(randomized_text_input_list)

        with torch.no_grad():
            outputs = textattack.shared.utils.batch_model_predict(
                self._model_predict, ids, batch_size=self.batch_size
            )
        
        label_nums = outputs.shape[1]
        ensemble_logits_for_each_input = np.split(outputs, indices_or_sections=len(text_input_list), axis=0)
        logits_list = []
        for logits in ensemble_logits_for_each_input:
            if self.ensemble_method == 'votes':
                probs = np.bincount(np.argmax(logits, axis=-1), minlength=label_nums) / self.ensemble
                logits_list.append(np.expand_dims(probs, axis=0))
            else:
                probs = normalize(logits, axis=1)
                probs = np.mean(probs, axis=0, keepdims=True)
                logits_list.append(probs)

        outputs = np.concatenate(logits_list, axis=0)
        return outputs

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        ids = self.encode([text_input])
        predictions = self._model_predict(ids)

        model_device = next(self.model.parameters()).device
        input_dict = {k: [_dict[k] for _dict in ids] for k in ids[0]}
        input_dict = {
            k: torch.tensor(v).to(model_device) for k, v in input_dict.items()
        }
        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": ids[0]["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(x)["input_ids"])
            for x in inputs
        ]
