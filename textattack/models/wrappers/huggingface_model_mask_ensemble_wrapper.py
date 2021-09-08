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
from utils.mask import mask_sentence, mask_forbidden_index
from scipy.special import softmax
from sklearn.preprocessing import normalize
from utils.my_utils import build_forbidden_mask_words
from args import ProgramArgs
from torch import nn as nn
from transformers import PreTrainedTokenizer, AutoModelForMaskedLM, RobertaTokenizer


class HuggingFaceModelMaskEnsembleWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""
    def __init__(self, 
                args: ProgramArgs,
                model: nn.Module, 
                tokenizer: PreTrainedTokenizer, 
                batch_size: int = 300, 
                with_lm: bool = False):
        self.model = model.to(textattack.shared.utils.device)
        self.mask_token = tokenizer.mask_token
        self.ensemble = args.predict_ensemble
        if isinstance(tokenizer, transformers.PreTrainedTokenizer):
            tokenizer = textattack.models.tokenizers.AutoTokenizer(tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.mask_rate = args.sparse_mask_rate
        self.ensemble_method = args.ensemble_method
        self.max_seq_length = args.max_seq_length

        self.forbidden_words = None
        if args.keep_sentiment_word:
            self.forbidden_words = build_forbidden_mask_words(args.sentiment_path)
        self.masked_lm = None
        if args.with_lm:
            self.lm_tokenizer = tokenizer.tokenizer
            masked_lm = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
            self.masked_lm = masked_lm.to(textattack.shared.utils.device)
            self.masked_lm.eval()

        
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

    def mask_sentence_decrator(self, sentence:str) -> List[str]:
        forbidden_index = None
        random_probs = None
        if self.forbidden_words is not None:
            forbidden_index = mask_forbidden_index(sentence, self.forbidden_words)
        if self.masked_lm is not None:
            random_probs = self.mask_probs(sentence) 

        return mask_sentence(sentence, self.mask_rate, self.mask_token, self.ensemble, forbidden=forbidden_index, random_probs=random_probs)

    def mask_lm_loss(self, ids: torch.Tensor, pred: torch.Tensor, delete_special_tokens:bool=True) -> torch.Tensor:
        loss = torch.nn.functional.cross_entropy(pred, ids, reduction='none')
        if delete_special_tokens:
            loss[ids == self.lm_tokenizer.pad_token_id] = 0.0
            loss[ids == self.lm_tokenizer.sep_token_id] = 0.0
            loss[ids == self.lm_tokenizer.cls_token_id] = 0.0
        return loss

    def get_tokenizer_mapping_for_sentence(self, sentence: str) -> Tuple: 
        if isinstance(self.lm_tokenizer, RobertaTokenizer):
            sentence_tokens = sentence.split()
            enc_result = [self.lm_tokenizer.encode(sentence_tokens[0], add_special_tokens=False)]
            enc_result.extend([self.lm_tokenizer.encode(x, add_special_tokens=False, add_prefix_space=True) for x in sentence_tokens[1:]])
        else:
            enc_result = [self.lm_tokenizer.encode(x, add_special_tokens=False) for x in sentence.split()]
        desired_output = []
        idx = 1
        for token in enc_result:
            tokenoutput = []
            for _ in token:
                tokenoutput.append(idx)
                idx += 1
            desired_output.append(tokenoutput)
        return (enc_result, desired_output)

    def get_word_loss(self, indexes: List[int], losses: torch.Tensor) -> float:
        try:
            loss = []
            for index in indexes:
                if index <= self.max_seq_length - 2:
                    loss.append(losses[index].item())
                else:
                    loss.append(0.0)
            return np.mean(loss)
        except:
            return 0.0

    def mask_probs(self, sentence: str) -> List[float]:
        encodings = self.lm_tokenizer.encode_plus(sentence, truncation=True, max_length=self.max_seq_length, add_special_tokens=True, return_tensors='pt')
        encodings = {key: value.cuda() for key, value in encodings.items()}
        lm_logits = self.masked_lm(**encodings)[0]
        lm_losses = self.mask_lm_loss(encodings["input_ids"][0], lm_logits[0])
        _, mappings = self.get_tokenizer_mapping_for_sentence(sentence)
        mask_probs = [self.get_word_loss(mapping, lm_losses) for mapping in mappings]
        mask_probs = mask_probs / sum(mask_probs)
        return mask_probs

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        mask_text_input_list = []
        if isinstance(text_input_list[0], tuple) and len(text_input_list[0]) == 2:
            for text_input in text_input_list:
                mask_text_input = [(text_input[0], sentence) for sentence in self.mask_sentence_decrator(text_input[1])]
                mask_text_input_list.extend(mask_text_input)
        else:
            for text_input in text_input_list:
                mask_text_input_list.extend(self.mask_sentence_decrator(text_input))

        ids = self.encode(mask_text_input_list)

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
