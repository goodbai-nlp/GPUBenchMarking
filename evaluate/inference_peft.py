# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import json
import sys
import os
import math
import time
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
import torch.distributed as dist
from peft import PeftModel, PeftConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--max_memory",
        type=float,
        default=0.45,
        help="Maximum allowable GPU memory useage for each GPU",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="Path to test_file",
        required=True,
    )
    parser.add_argument(
        "--instruction",
        default="Generate the constituent tree for a given sentence.",
        type=str,
        help="input file",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Specify num of return sequences",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=400,
        help="Specify num of return sequences",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        default=False,
        help="whether to use low_cpu_mem_usage for inference",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        help="testset",
        required=True,
    )
    parser.add_argument("--bit_8", action="store_true", default=False)
    parser.add_argument(
        "--offload_state_dict",
        action="store_true",
        default=False,
        help="Whether to offload state dict (useful for very large LMs)",
    )
    parser.add_argument(
        "--offload_folder",
        type=str,
        default="resources/offload/",
        help="directory path for offloading",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="None",
        help="directory path for offloading",
    )
    parser.add_argument(
        "--input_key",
        type=str,
        default="sentence",
        help="input key of test set",
    )

    args = parser.parse_args()

    return args


def set_max_memory(args):
    n_gpus = torch.cuda.device_count()
    if args.max_memory and n_gpus > 1:
        logger.info("Infering max memory...")
        t = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
        # note, we use math.floor() as a conservative rounding method
        # to optimize the maximum batch size on multiple GPUs, we give the first GPU less memory
        # see max_memory at https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling
        max_memory = {
            i: (
                f"{math.floor(t*args.max_memory)}GiB"
                if i > 0
                else f"{math.floor(t*args.max_memory*0.2)}GiB"
            )
            for i in range(n_gpus)
        }
        # max_memory['cpu'] = '400GiB' # may need to lower this depending on hardware
        logger.info(f"Set maximum memory: {max_memory}")
        return max_memory
    else:
        return None


def prompt_eval(args, model, tokenizer, prompts):
    all_res = []
    with torch.inference_mode():
        for prompt in tqdm(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(torch.cuda.current_device())
            # print("input_ids:", inputs.input_ids)
            generate_ids = model.generate(
                input_ids=inputs.input_ids,
                num_beams=args.num_beams,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                max_new_tokens=args.max_new_tokens,
            )
            # print("generate_ids:", generate_ids)
            ori_output = tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # print(f"{ori_output}")
            new_ids = generate_ids[:, inputs.input_ids.size(1):]
            # print("new_ids:", new_ids)
            output = tokenizer.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            all_res += output
            # print(f"{output}")
            
    return all_res


def create_prompt(data, args):
    if args.prompt_template == "None":
        prompts = [itm[args.input_key] for itm in data]
    elif args.prompt_template == "pubmedqa":
        prompts = [f"{sample['context'].rstrip()}\nQuestion:\n{sample['QUESTION']}\nPlease respond with yes, no or maybe. The answer to the question is:" for sample in data]
    elif args.prompt_template == "supervised":
        prompts = [
            f"Human:\n{args.instruction} {sample[args.input_key]}\nAssistant:\n"
            for sample in data
        ]
    elif args.prompt_template == "vicuna":
        prompts = [
            f"USER: {args.instruction}{sample[args.input_key]} ASSISTANT:"
            for sample in data
        ]
    elif args.prompt_template == "gpt":
        prompts = [
            f"Human:\n{args.instruction}{sample[args.input_key]}\nAssistant:\n"
            for sample in data
        ]
    elif args.prompt_template == "one_shot":
        prompts = [
            f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n### Human: Got any creative ideas for a 10 year old’s birthday?\n### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:\n1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\n3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.\n4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.\n5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.\n6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.\n7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.\n8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.\nRemember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!\n### Human: {sample[args.input_key]}\n### Assistant:"
            for sample in data
        ]
    elif args.prompt_template == "one_shot_sim":
        prompts = [
            f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n### Human: Got any creative ideas for a 10 year old’s birthday?\n### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:\n1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\nRemember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!\n### Human: {sample[args.input_key]}\n### Assistant:"
            for sample in data
        ]
    else:
        print("Invalid Prompt template, exit ...")
        exit()
    return prompts


local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = torch.cuda.device_count()
rank = local_rank


def main():
    args = parse_args()
    start_time = time.time()
    config = PeftConfig.from_pretrained(args.model_name_or_path)
    print(f"Loading tokenizer from {args.base_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, fast_tokenizer=False, add_eos_token=False)
    
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, device_map="auto")
    model = PeftModel.from_pretrained(base_model, args.model_name_or_path, device_map= "auto")
    end_time = time.time()
    logger.info(f"Loaded model {args.model_name_or_path} in {end_time - start_time:.4f} seconds")
    model.print_trainable_parameters()
    logger.info(f"Model parameters: {model.num_parameters():,} / footprint: {model.get_memory_footprint() / (1024*1024*1024):.2f} GB")

    with open(args.test_file, "r", encoding="utf-8") as fin:
        data = [json.loads(line.strip()) for line in fin]
        prompts = create_prompt(data, args)

    print("Start inference ...")

    pred_res = prompt_eval(args, model, tokenizer, prompts)
    pred_res_cleaned = [itm.replace("\n", " ") for itm in pred_res]
    out_prefix = args.test_file.split("/")[-1]
    with open(
        f"{args.model_name_or_path}/{args.out_prefix}_{out_prefix}_pred_peft", "w", encoding="utf-8"
    ) as fout:
        fout.write("\n".join(pred_res_cleaned) + "\n")


if __name__ == "__main__":
    main()
