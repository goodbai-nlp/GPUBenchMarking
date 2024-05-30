# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from dataclasses import dataclass
from datasets import load_dataset
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from torch.utils.data import Subset
import re


def padding_func(
    features,
    padding_side="right",
    pad_token_id=1,
    key="label",
    pad_to_multiple_of=1,
    max_length=None,
):
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features)
    if pad_to_multiple_of > 1:
        if max_length is not None:
            max_label_length = min(
                max_length,
                (max_label_length + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of,
            )
        else:
            max_label_length = (
                (max_label_length + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )

    for feature in features:
        remainder = [pad_token_id] * (max_label_length - len(feature[key]))
        feature[key] = (
            feature[key] + remainder if padding_side == "right" else remainder + feature[key]
        )
    return


@dataclass
class DataCollatorForCausalLM:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if "labels" in features[0].keys():
            padding_func(
                features,
                padding_side=self.tokenizer.padding_side,
                pad_token_id=self.label_pad_token_id,
                key="labels",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()
        return batch
    

@dataclass
class JointDataCollatorForCausalLM:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if "labels" in features[0].keys():
            padding_func(
                features,
                padding_side=self.tokenizer.padding_side,
                pad_token_id=self.label_pad_token_id,
                key="labels",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        if "input_view2_ids" in features[0].keys():
            padding_func(
                features,
                padding_side=self.tokenizer.padding_side,
                pad_token_id=self.tokenizer.pad_token_id,
                key="input_view2_ids",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()
        
        if "view2_labels" not in batch:
            batch["view2_labels"] = batch["input_view2_ids"].clone()
        return batch


def get_raw_dataset(dataset_name, output_path, seed, local_rank, out_format="bracket"):
    if dataset_name.endswith("stanford-alpaca"):
        return StanfordAlpacaDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("Dahoas/rm-static"):
        return DahoasRmstaticDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("Dahoas/full-hh-rlhf"):
        return DahoasFullhhrlhfDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("Dahoas/synthetic-instruct-gptj-pairwise"):
        return DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif dataset_name.endswith("yitingxie/rlhf-reward-datasets"):
        return YitingxieRlhfrewarddatasetsDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("openai/webgpt_comparisons"):
        return OpenaiWebgptcomparisonsDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("stanfordnlp/SHP"):
        return StanfordnlpSHPDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("wangrui6/Zhihu-KOL"):
        return Wangrui6ZhihuKOLDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("Cohere/miracl-zh-queries-22-12"):
        return CohereMiraclzhqueries2212Dataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("Hello-SimpleAI/HC3-Chinese"):
        return HelloSimpleAIHC3ChineseDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("mkqa-Chinese"):
        return MkqaChineseDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("mkqa-Japanese"):
        return MkqaJapaneseDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("Cohere/miracl-ja-queries-22-12"):
        return CohereMiracljaqueries2212Dataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("lmqg/qg_jaquad"):
        return LmqgQgjaquadDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("lmqg/qag_jaquad"):
        return LmqgQagjaquadDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("webnlg"):
        return WebnlgDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("wsj"):
        return ConParsingDataset(output_path, seed, local_rank, dataset_name, out_format)
    elif dataset_name.endswith("wsj-transition"):
        return ConParsingTransitionDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("wsj-joint"):
        return NERConParsingDataset(output_path, seed, local_rank, dataset_name, out_format)
    elif dataset_name.endswith("domain-data"):
        return RawTextDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("pubmed-abs"):
        return PubMedDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("trucated-pubmedqa"):
        return TrucatedPubMedQADataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("pubmedqa"):
        return PubMedQADataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("-amr"):
        return AMRDataset(output_path, seed, local_rank, dataset_name)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):
    def __init__(self, output_path, seed, local_rank, data_path, eos_token=""):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        # self.eos_token = "<|endoftext|>"
        self.eos_token = eos_token
        self.raw_datasets = load_dataset(data_path)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


# English dataset
class DahoasRmstaticDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["prompt"]

    def get_chosen(self, sample):
        return sample["chosen"]

    def get_rejected(self, sample):
        return sample["rejected"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"] + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"] + sample["rejected"]

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"prompt": prompt, "chosen": chosen, "reject": reject}) for prompt, chosen, reject in zip(samples["prompt"], samples["chosen"], samples["rejected"])
        ]
        return {"text": input_text}
    

class StanfordAlpacaDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "stanford-alpaca"
        self.dataset_name_clean = "stanford-alpaca"
        self.input_key = "input"
        self.output_key = "output"
        print("Loaded dataset", self.raw_datasets)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return f" Human:\n{sample['instruction']} {sample['input']}\nAssistant:\n"

    def get_chosen(self, sample):
        return sample["output"]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        # return sample["instruction"] + sample["input"] + sample["output"]
        return f" Human:\n{sample['instruction']} {sample['input']}\nAssistant:\n{sample['output']}"

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({self.input_key: input, self.output_key: output, "instruction": instruction})
            for instruction, input, output in zip(
                samples["instruction"], samples[self.input_key], samples[self.output_key]
            )
        ]
        input_prompt = [
            self.get_prompt({self.input_key: input,"instruction": instruction})
            for instruction, input in zip(
                samples["instruction"], samples[self.input_key]
            )
        ]
        return {"text": input_text, "prompt": input_prompt}


class PubMedQADataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "pubmedqa"
        self.dataset_name_clean = "pubmedqa"
        self.input_key = "input"
        self.output_key = "final_decision"
        print("Loaded dataset", self.raw_datasets)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        assert len(sample["CONTEXTS"]) == len(sample["LABELS"])
        context = " ".join([f'{topic.lower()}: {con}' for con, topic in zip(sample["CONTEXTS"], sample["LABELS"])][:250])
        return f"CONTEXT:\n{context}\nQUESTION:\n{sample['QUESTION']}\nANSWER:\n{sample['LONG_ANSWER']}:\nGiven the CONTEXT, QUESTION and ANSWER, judge whether the provided ANSWER correctly addresses the QUESTION under the given CONTEXT. Please output yes, no or maybe. The output is:"

    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return f"{self.get_prompt(sample)}{sample[self.output_key]}"

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"CONTEXTS": context, "QUESTION": question, "LABELS": label, "LONG_ANSWER": answer})
            for context, question, label, answer in zip(
                samples["CONTEXTS"], samples["QUESTION"], samples["LABELS"], samples["LONG_ANSWER"]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"CONTEXTS": context, "QUESTION": question, "LABELS": label, "LONG_ANSWER": answer, "final_decision": decision})
            for context, question, label, answer, decision in zip(
                samples["CONTEXTS"], samples["QUESTION"], samples["LABELS"], samples["LONG_ANSWER"], samples["final_decision"]
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class TrucatedPubMedQADataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "trucated-pubmedqa"
        self.dataset_name_clean = "trucated-pubmedqa"
        self.input_key = "context"
        self.output_key = "final_decision"
        print("Loaded dataset", self.raw_datasets)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return f"{sample['context'].rstrip()}\nQuestion:\n{sample['QUESTION']}\nPlease respond with yes, no or maybe. The answer to the question is:"

    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return f"{self.get_prompt(sample)}{sample[self.output_key]}"

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"context": context, "QUESTION": question})
            for context, question in zip(
                samples["context"], samples["QUESTION"]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"context": context, "QUESTION": question, "final_decision": decision})
            for context, question, decision in zip(
                samples["context"], samples["QUESTION"], samples["final_decision"]
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class PubMedDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "pubmed-abs"
        self.dataset_name_clean = "pubmed-abs"
        print("Loaded dataset", self.raw_datasets)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["abstract"]

    def get_chosen(self, sample):
        return sample["abstract"]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include chosen response.")

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_text = samples["abstract"]
        return {"text": input_text}
    

class RawTextDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "rawtext"
        self.dataset_name_clean = "rawtext"
        print("Loaded dataset", self.raw_datasets)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["text"]

    def get_chosen(self, sample):
        return sample["text"]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include chosen response.")

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        return {"text": samples["text"]}


class WebnlgDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "webnlg"
        self.dataset_name_clean = "webnlg"
        self.raw_datasets = load_dataset(
            data_path,
            data_files={
                "train": f"{data_path}/train.jsonl",
                "validation": f"{data_path}/val.jsonl",
                "test": f"{data_path}/test.jsonl",
            },
        )
        self.instruction = "Generate a descriptive text for the given knowledge graph."

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return f"Human:\n{self.instruction} {sample['src']}\nAssistant:\n"

    def get_chosen(self, sample):
        return sample["tgt"]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return f"{self.get_prompt(sample)}{sample['tgt']}"
    
    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None
    
    def process_function(self, samples):
        input_prompt = [
            self.get_prompt_and_chosen({"src": input, "tgt": output})
            for input, output in zip(
                samples["src"], samples["tgt"]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"src": input, "tgt": output})
            for input, output in zip(
                samples["src"], samples["tgt"]
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class AMRDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path, task="amr2text"):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "amrdataset"
        self.dataset_name_clean = "amrdataset"
        self.raw_datasets = load_dataset(
            data_path,
            data_files={
                "train": f"{data_path}/train.jsonl",
                "valid": f"{data_path}/val.jsonl",
                "test": f"{data_path}/test.jsonl",
            },
        )
        self.task = task
        assert self.task in ("amrparsing", "amr2text")
        if self.task == "amr2text":
            self.instruction = "Generate a descriptive text for the given abstract meaning representation graph."
        else:
            self.instruction = "Generate the AMR graph for the given input text."
            
    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["valid"]

    def get_prompt(self, sample):
        if self.task == "amr2text":
            return f"Human:\n{self.instruction} {sample['amr']}\nAssistant:\n"
        else:
            return f"Human:\n{self.instruction} {sample['sentence']}\nAssistant:\n"

    def get_chosen(self, sample):
        return sample["sentence"] if self.task == "amr2text" else sample["amr"]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        if self.task == "amr2text":
            return f"{self.get_prompt(sample)}{sample['sentence']}"
        else:
            return f"{self.get_prompt(sample)}{sample['amr']}"

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"amr": input, "sentence": output})
            for input, output in zip(
                samples["amr"], samples["sentence"]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"amr": input, "sentence": output})
            for input, output in zip(
                samples["amr"], samples["sentence"]
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class ConParsingDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path, out_format="bracket"):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "wsj"
        self.dataset_name_clean = "wsj"
        self.output_key = out_format
        self.input_key = "sentence"
        assert self.output_key in ["bracket", "SoR"]
        self.instruction = "Generate the constituent tree for a given sentence."

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        # return self.instruction + sample["sentence"]
        return f" Human: {self.instruction} {sample['sentence']} Assistant: "

    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        # return self.instruction + sample["sentence"] + sample[self.output_key]
        return (
            f" Human: {self.instruction} {sample['sentence']} Assistant: {sample[self.output_key]}"
        )

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"sentence": input, f"{self.output_key}": output})
            for input, output in zip(
                samples[self.input_key], samples[self.output_key]
            )
        ]
        return {"text": input_text}


class ConParsingTransitionDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "wsj"
        self.dataset_name_clean = "wsj"
        self.output_key = "SoR"
        self.input_key = "sentence"
        self.instruction = "Generate the constituent tree for a given sentence."

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        # return self.instruction + sample["sentence"]
        return f" Human: {self.instruction} {sample['sentence']} Assistant: " 

    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        # return self.instruction + sample["sentence"] + sample[self.output_key]
        return (
            f" Human: {self.instruction} {sample['sentence']} {sample['pos']} Assistant: {sample[self.output_key]}"
        )

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"sentence": input, "SoR": output, "pos": pos})
            for input, output, pos in zip(
                samples[self.input_key], samples[self.output_key], samples["pos"]
            )
        ]
        return {"text": input_text}


class NERConParsingDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path, out_format="bracket"):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "wsj"
        self.dataset_name_clean = "wsj"
        self.output_key = out_format
        self.input_key = "sentence"
        assert self.output_key in ["bracket", "SoR"]
        self.instruction = "Generate the constituent tree for a given sentence."
        self.ner_instruction = "Mark all named entities in the given sentence."

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        # return self.instruction + sample["sentence"]
        return f" Human: {self.instruction} {sample['sentence']} Assistant: "
    
    def get_ner_prompt(self, sample):
        # return self.instruction + sample["sentence"]
        return f" Human: {self.ner_instruction} {sample['sentence']} Assistant: " 

    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        # return self.instruction + sample["sentence"] + sample[self.output_key]
        return (
            f" Human: {self.instruction} {sample['sentence']} Assistant: {sample[self.output_key]}"
        )
        
    def get_ner_prompt_and_chosen(self, sample):
        # return self.instruction + sample["sentence"] + sample[self.output_key]
        return (
            f" Human: {self.ner_instruction} {sample['sentence']} Assistant: {sample['ner']}"
        )

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"sentence": input, f"{self.output_key}": output})
            for input, output in zip(
                samples[self.input_key], samples[self.output_key]
            )
        ]
        input_ner_text = [
            self.get_ner_prompt_and_chosen({"sentence": input, "ner": output})
            for input, output in zip(
                samples[self.input_key], samples['ner']
            )
        ]
        return {
            "text": input_text,
            "ner_text": input_ner_text,
        }


# English dataset
class DahoasFullhhrlhfDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "Dahoas/full-hh-rlhf"
        self.dataset_name_clean = "Dahoas_full_hh_rlhf"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["prompt"]

    def get_chosen(self, sample):
        return sample["chosen"]

    def get_rejected(self, sample):
        return sample["rejected"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"] + sample["chosen"] + self.eos_token

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"] + sample["rejected"] + self.eos_token

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"prompt": prompt, "chosen": chosen})
            for prompt, chosen in zip(
                samples["prompt"], samples["chosen"]
            )
        ]
        return {"text": input_text}


# English dataset
class DahoasSyntheticinstructgptjpairwiseDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "Dahoas/synthetic-instruct-gptj-pairwise"
        self.dataset_name_clean = "Dahoas_synthetic_instruct_gptj_pairwise"

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_prompt(self, sample):
        return " Human: " + sample["prompt"] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample["chosen"]

    def get_rejected(self, sample):
        return " " + sample["rejected"]

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample["prompt"] + " Assistant: " + sample["chosen"] + self.eos_token

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample["prompt"] + " Assistant: " + sample["rejected"] + self.eos_token

    def process_function(self, samples):
        input_text = [
            {"prompt": prompt, "chosen": chosen}
            for prompt, chosen in zip(
                samples["prompt"], samples["chosen"]
            )
        ]
        return {"text": input_text}


# English dataset
class YitingxieRlhfrewarddatasetsDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "yitingxie/rlhf-reward-datasets"
        self.dataset_name_clean = "yitingxie_rlhf_reward_datasets"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["prompt"] + "Assistant:"

    def get_chosen(self, sample):
        return sample["chosen"].split("Assistant:")[-1]

    def get_rejected(self, sample):
        return sample["rejected"].split("Assistant:")[-1]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"] + sample["chosen"] + self.eos_token

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"] + sample["rejected"] + self.eos_token
    
    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"prompt": prompt, "chosen": chosen})
            for prompt, chosen in zip(
                samples["prompt"], samples["chosen"]
            )
        ]
        return {"text": input_text}


# English dataset
class OpenaiWebgptcomparisonsDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "openai/webgpt_comparisons"
        self.dataset_name_clean = "openai_webgpt_comparisons"

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_prompt(self, sample):
        return " Human: " + sample["question"]["full_text"] + " Assistant:"

    def get_chosen(self, sample):
        if float(sample["score_0"]) >= float(sample["score_1"]):
            response = sample["answer_0"]
        else:
            response = sample["answer_1"]
        # This data has citation square brackets and numbers (e.g., "[1]").
        # Right now we are not doing browser-assisted finetuning, thus we
        # remove these citations to avoid confusing the model.
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_rejected(self, sample):
        if float(sample["score_0"]) < float(sample["score_1"]):
            response = sample["answer_0"]
        else:
            response = sample["answer_1"]
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if float(sample["score_0"]) >= float(sample["score_1"]):
            response = sample["answer_0"]
        else:
            response = sample["answer_1"]
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample["question"]["full_text"] + " Assistant: " + response + self.eos_token

    def get_prompt_and_rejected(self, sample):
        if float(sample["score_0"]) < float(sample["score_1"]):
            response = sample["answer_0"]
        else:
            response = sample["answer_1"]
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample["question"]["full_text"] + " Assistant: " + response + self.eos_token

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"prompt": question, "answer_0": ans0, "answer_1": ans1, "score_0": score0, "score_1": score1})
            for question, ans0, ans1, score0, score1 in zip(
                samples["question"], samples["answer_0"], samples["answer_1"], samples["score_0"], samples["score_1"]
            )
        ]
        return {"text": input_text}
    
    
# English dataset
class StanfordnlpSHPDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "stanfordnlp/SHP"
        self.dataset_name_clean = "stanfordnlp_SHP"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample["history"] + " Assistant:"

    def get_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " " + response

    def get_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " Human: " + sample["history"] + " Assistant: " + response + self.eos_token

    def get_prompt_and_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " Human: " + sample["history"] + " Assistant: " + response + self.eos_token

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"history": his, "labels": label, "human_ref_A": ref1, "human_ref_B": ref2})
            for his, label, ref1, ref2 in zip(
                samples["history"], samples["labels"], samples["human_ref_A"], samples["human_ref_B"]
            )
        ]
        return {"text": input_text}


# Chinese dataset
class Wangrui6ZhihuKOLDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "wangrui6/Zhihu-KOL"
        self.dataset_name_clean = "wangrui6_Zhihu_KOL"

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_prompt(self, sample):
        if sample["INSTRUCTION"] is not None:
            return " Human: " + sample["INSTRUCTION"] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample["RESPONSE"] is not None:
            return " " + sample["RESPONSE"]
        return None

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        if sample["INSTRUCTION"] is not None and sample["RESPONSE"] is not None:
            return " Human: " + sample["INSTRUCTION"] + " Assistant: " + sample["RESPONSE"] + self.eos_token
        return None

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.") + self.eos_token
        return None

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"INSTRUCTION": prompt, "RESPONSE": chosen})
            for prompt, chosen in zip(
                samples["INSTRUCTION"], samples["RESPONSE"]
            )
        ]
        return {"text": input_text}
    

# Chinese dataset
class CohereMiraclzhqueries2212Dataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "Cohere/miracl-zh-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_zh_queries_22_12"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample["query"] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample["positive_passages"][0]["text"]

    def get_rejected(self, sample):
        return " " + sample["negative_passages"][0]["text"]

    def get_prompt_and_chosen(self, sample):
        return (
            " Human: " + sample["query"] + " Assistant: " + sample["positive_passages"][0]["text"] + self.eos_token
        )

    def get_prompt_and_rejected(self, sample):
        return (
            " Human: " + sample["query"] + " Assistant: " + sample["negative_passages"][0]["text"] + self.eos_token
    
        )

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"query": prompt, "positive_passages": chosen})
            for prompt, chosen in zip(
                samples["query"], samples["positive_passages"]
            )
        ]
        return {"text": input_text}
    

# Chinese dataset
class HelloSimpleAIHC3ChineseDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "Hello-SimpleAI/HC3-Chinese"
        self.dataset_name_clean = "Hello_SimpleAI_HC3_Chinese"

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_prompt(self, sample):
        if sample["question"] is not None:
            return " Human: " + sample["question"] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample["human_answers"][0] is not None:
            return " " + sample["human_answers"][0]
        return None

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        if sample["question"] is not None and sample["human_answers"][0] is not None:
            return " Human: " + sample["question"] + " Assistant: " + sample["human_answers"][0] + self.eos_token
        return None

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"question": prompt, "human_answers": chosen})
            for prompt, chosen in zip(
                samples["question"], samples["human_answers"]
            )
        ]
        return {"text": input_text}
    

# Chinese dataset
class MkqaChineseDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "mkqa-Chinese"
        self.dataset_name_clean = "mkqa"

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_prompt(self, sample):
        if sample["queries"]["zh_cn"] is not None:
            return " Human: " + sample["queries"]["zh_cn"] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample["answers"]["zh_cn"][0]["text"] is not None:
            return " " + sample["answers"]["zh_cn"][0]["text"]
        return None

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        if (
            sample["queries"]["zh_cn"] is not None
            and sample["answers"]["zh_cn"][0]["text"] is not None
        ):
            return (
                " Human: "
                + sample["queries"]["zh_cn"]
                + " Assistant: "
                + sample["answers"]["zh_cn"][0]["text"] + self.eos_token
            )
        return None

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None
    
    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"queries": prompt, "answers": chosen})
            for prompt, chosen in zip(
                samples["queries"], samples["answers"]
            )
        ]
        return {"text": input_text}
    

# Japanese dataset
class MkqaJapaneseDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "mkqa-Japanese"
        self.dataset_name_clean = "mkqa"

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_prompt(self, sample):
        if sample["queries"]["ja"] is not None:
            return " Human: " + sample["queries"]["ja"] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample["answers"]["ja"][0]["text"] is not None:
            return " " + sample["answers"]["ja"][0]["text"]
        return None

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        if sample["queries"]["ja"] is not None and sample["answers"]["ja"][0]["text"] is not None:
            return (
                " Human: "
                + sample["queries"]["ja"]
                + " Assistant: "
                + sample["answers"]["ja"][0]["text"]
                + self.eos_token
            )
        return None

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"queries": prompt, "answers": chosen})
            for prompt, chosen in zip(
                samples["queries"], samples["answers"]
            )
        ]
        return {"text": input_text}


# Japanese dataset
class CohereMiracljaqueries2212Dataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "Cohere/miracl-ja-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_ja_queries_22_12"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample["query"] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample["positive_passages"][0]["text"]

    def get_rejected(self, sample):
        return " " + sample["negative_passages"][0]["text"]

    def get_prompt_and_chosen(self, sample):
        return (
            " Human: " + sample["query"] + " Assistant: " + sample["positive_passages"][0]["text"] + self.eos_token
        )

    def get_prompt_and_rejected(self, sample):
        return (
            " Human: " + sample["query"] + " Assistant: " + sample["negative_passages"][0]["text"] + self.eos_token
        )

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"query": prompt, "positive_passages": chosen})
            for prompt, chosen in zip(
                samples["query"], samples["positive_passages"]
            )
        ]
        return {"text": input_text}
    

# Japanese dataset
class LmqgQgjaquadDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "lmqg/qg_jaquad"
        self.dataset_name_clean = "lmqg_qg_jaquad"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample["question"] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample["sentence"]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample["question"] + " Assistant: " + sample["sentence"] + self.eos_token

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None
    
    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"question": prompt, "sentence": chosen})
            for prompt, chosen in zip(
                samples["question"], samples["sentence"]
            )
        ]
        return {"text": input_text}
    

# Japanese dataset
class LmqgQagjaquadDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "lmqg/qag_jaquad"
        self.dataset_name_clean = "lmqg_qag_jaquad"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample["questions"][0] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample["paragraph"]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample["questions"][0] + " Assistant: " + sample["paragraph"] + self.eos_token

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_text = [
            self.get_prompt_and_chosen({"questions": prompt, "paragraph": chosen})
            for prompt, chosen in zip(
                samples["questions"], samples["paragraph"]
            )
        ]
        return {"text": input_text}