# Used to calculate the log probability of a given conversation. 
# Modified from https://github.com/kanishkamisra/minicons/blob/master/minicons/scorer.py


from typing import (
    Iterable,
    Union,
    List,
    Optional,
    Callable,
    Tuple,
)

import torch
import warnings

from collections import defaultdict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
)

def target_tokens(
                  tokenizer: AutoTokenizer,
                    contents: List[str],
                 ) -> List[BatchEncoding]:
    '''
    Calculate the target tokens of the contents.
    '''
    tokens = []
    for i in contents:
        encoded = tokenizer.encode(i, add_special_tokens=False)
        tokens.append(encoded)
    return tokens

def batch_wise_logprobs(logprobs, ids, reduction):
    batch_wise = [torch.stack(token_wise).T for token_wise in list(zip(*logprobs))]
    batch_labels = []
    for batch in batch_wise:
        batch_labels.append(torch.stack([reduction(y[i]) for i, y in zip(ids, batch)]))

    return torch.stack(batch_labels)

class LMScorer:
    """
    Base LM scorer class intended to store models and tokenizers along
    with methods to facilitate the analysis of language model output scores.
    """

    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        device: Optional[str] = "cpu",
        tokenizer=None,
        **kwargs,
    ) -> None:
        """
        :param model: should be path to a model (.pt or .bin file) stored
            locally, or name of a pretrained model stored on the Huggingface
            Model Hub, or a model (torch.nn.Module) that have the same
            signature as the corresponding Huggingface model (see the subclass
            for details).
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        :param tokenizer: if provided, use this tokenizer.
        """
        if tokenizer is not None:
            if isinstance(tokenizer, str):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
            else:
                self.tokenizer = tokenizer
        elif isinstance(model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        else:
            raise Exception("Must provide either model name or tokenizer.")
        self.device = device
        self.vocab = defaultdict(list)
        # {self.vocab[x.strip()].append(i) for x, i in [(self.tokenizer.decode([i]), i) for i in range(self.tokenizer.vocab_size)]}
        for i in range(self.tokenizer.vocab_size):
            decoded = [(self.tokenizer.decode(i), i)]
            for x, j in decoded:
                self.vocab[x.strip()].append(j)

    def conditional_score(
        self,
        target_tokens: List[BatchEncoding],
        stimuli: BatchEncoding,
        separator: str = " ",
        reduction: Callable = lambda x: x.mean(0).item(),
        prob: bool = False,
        base_two: bool = False,
        bow_correction: bool = False,
        **kw,
    ) -> List[float]:
        """
        Pooled estimates of sequence log probabilities (or some modification of it), given a prefix. Pooling is usually done using a function that is passed to the method.

        :param target_tokens: the target tokens of the contents.
        :type target_tokens: List[BatchEncoding]
        :param stimuli: Full encoded conversation.
        :type stimuli: ``BatchEncoding``
        :param reduction: Reduction function, is selected to be
            ``lambda x: x.mean(0).item()`` by default, which stands for the avg. log-probability per token for each sequence in the batch.
        :type reduction: Callable
        :param kw: model-specific keyword arguments to pass to the `prepare_text` function
        :return: List of floats specifying the desired score for the stimuli part of the input, e.g., P(stimuli | preamble).
        :rtype: ``List[float]``
        """
        primed = (stimuli, target_tokens)

        result = self.compute_stats(
            primed,
            rank=False,
            base_two=base_two,
            prob=prob,
            bow_correction=bow_correction,
            return_tensors=True,
        )
        logprob = result
        reduced = list(map(reduction, logprob))

        return reduced

class IncrementalLMScorer(LMScorer):
    """
    Class for Autoregressive or Incremental (or left-to-right) language models such as GPT2, etc.

    :param model: should be path to a model (.pt or .bin file) stored locally,
        or name of a pretrained model stored on the Huggingface Model Hub, or
        a model (torch.nn.Module) that have the same signature as a
        Huggingface model obtained from `AutoModelForCausalLM`. In the last
        case, a corresponding tokenizer must also be provided.
    :param device: device type that the model should be loaded on,
        options: `cpu or cuda:{0, 1, ...}`
    :type device: str, optional
    :param tokenizer: if provided, use this tokenizer.
    """

    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        device: Optional[str] = "cpu",
        tokenizer=None,
        **kwargs,
    ) -> None:
        """
        :param model: should be path to a model (.pt or .bin file) stored
            locally, or name of a pretrained model stored on the Huggingface
            Model Hub, or a model (torch.nn.Module) that have the same
            signature as a Huggingface model obtained from
            `AutoModelForCausalLM`. In the last case, a corresponding tokenizer
            must also be provided.
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        :param tokenizer: if provided, use this tokenizer.
        """
        super(IncrementalLMScorer, self).__init__(
            model, device=device, tokenizer=tokenizer, **kwargs
        )

        if isinstance(model, str):
            if self.device == "auto":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model, device_map=self.device, return_dict=True, **kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model, return_dict=True, **kwargs
                )
        else:
            self.model = model

        if self.device != "auto":
            self.model.to(self.device)

        # define CLS and SEP tokens
        if self.tokenizer.pad_token is None:
            if tokenizer is not None:
                warnings.warn(
                    "tokenizer is changed by adding pad_token_id to the tokenizer."
                )
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": ["<pad>"]}
                )
                self.tokenizer.pad_token = "<pad>"
                self.model.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.padding_side == "left":
            self.tokenizer.padding_side = "right"

        if isinstance(model, str):
            self.model.eval()

        self.padding_side = self.tokenizer.padding_side

        self.padding_side = self.tokenizer.padding_side
        self.pad_token_id = self.tokenizer.pad_token_id

        # bow_subtokens, but only if the model has a
        try:
            self.bow_symbol = self.tokenizer.convert_ids_to_tokens(
                self.tokenizer(" ", add_special_tokens=False).input_ids[0]
            )[
                0
            ]  # sometimes this trick returns <bow_symbol><space>, in models like llava
        except:
            self.bow_symbol = None
        if (
            self.bow_symbol == self.tokenizer.bos_token
            or self.bow_symbol is None
            or self.bow_symbol == self.tokenizer.eos_token
        ):
            self.is_bow_tokenizer = False
        else:
            self.is_bow_tokenizer = True

        if self.is_bow_tokenizer:
            self.bow_subwords = defaultdict(lambda: False)
            for word, idx in self.tokenizer.get_vocab().items():
                if word[0] == self.bow_symbol:
                    self.bow_subwords[idx] = True
                else:
                    self.bow_subwords[idx] = False

            # for cases where the model has added tokens beyond the ones it comes with
            for idx, details in self.tokenizer.added_tokens_decoder.items():
                if details.lstrip == True:
                    self.bow_subwords[idx] = True

            self.bow_subwords = dict(self.bow_subwords)
            self.bow_subword_idx = [k for k, v in self.bow_subwords.items() if v]

    def compute_stats(
        self,
        batch: Iterable,
        rank: bool = False,
        prob: bool = False,
        base_two: bool = False,
        return_tensors: bool = False,
        bow_correction: bool = False,
    ) -> Union[Tuple[List[float], List[float]], List[float]]:
        """
        Primary computational method that processes a batch of prepared sentences and returns per-token scores for each sentence. By default, returns log-probabilities.

        :param ``Iterable`` batch: batched input as processed by ``prepare_text`` or ``prime_text``.
        :param ``bool`` rank: whether the model should also return ranks per word (based on the conditional log-probability of the word in context).
        :param ``bool`` prob: whether the model should return probabilities instead of log-probabilities. Can only be `True` when `base_two` is `False`.
        :param ``bool`` base_two: whether the base of the log should be 2 (usually preferred when reporting results in bits). Can only be `True` when `prob` is `False`.
        :param ``bool`` return_tensors: whether the model should return scores as a list of tensors instead of a list of lists. This is important in some other convenient methods used in the package.
        :param ``bool'' bow_correction: whether to apply the beginning of word correction, as pointed out in Pimentel and Meister (2024) and Oh and Schuler (2024).

        :return: Either a tuple of lists, each containing probabilities and ranks per token in each sentence passed in the input.
        :rtype: ``Union[Tuple[List[float], List[int]], List[float]]``
        """
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        encoded, target_tokens = batch
        offsets = [len(target_token) for target_token in target_tokens]
        ids = encoded

        effective_ids = target_tokens
        logits = []
        for id in ids:
            with torch.no_grad():
                logit = self.model(**id).logits.detach()
                logits.append(logit)

        ## Set up storage variables
        scores = []

        for logit, idx, offset in zip(logits, effective_ids, offsets):

            query_ids = idx
            logit = logit.squeeze(0)  # remove the batch dimension
            logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)
            
            actual_logprob_distribution = logprob_distribution[-offset-1:-1]  # get the log probability of the target tokens

            score = actual_logprob_distribution[
                torch.arange(offset), query_ids
            ]

            if base_two:
                """
                Log_2(X) = log_e(X)/log_e(2) (broadcasted)
                """
                score = score / torch.tensor(2).log()
            else:
                if prob:
                    score = score.exp()
                else:
                    score = score

            scores.append(score)

        if not return_tensors:
            # scores = [torch.tensor(l).detach() for l in scores]
            scores = [s.tolist() for s in scores]


        return scores