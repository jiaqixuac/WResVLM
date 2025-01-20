"""
https://github.com/Q-Future/Q-Instruct/blob/main/eval_scripts/llava_v1.5/eval_image_quality.py
https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/run_llava.py

python tools/rate_scripts/llava/eval_image_quality.py --mode cap-one --prompt-mode visibility
python tools/rate_scripts/llava/eval_image_quality.py --mode pseudo-labels --prompt-mode quality

python tools/rate_scripts/llava/eval_image_quality.py --mode pseudo-labels \
--model-path liuhaotian/llava-v1.6-mistral-7b --prompt-mode visibility
python tools/rate_scripts/llava/eval_image_quality.py --mode pseudo-labels \
--model-path liuhaotian/llava-v1.6-vicuna-13b --prompt-mode visibility
python tools/rate_scripts/llava/eval_image_quality.py --mode pseudo-labels \
--model-path liuhaotian/llava-v1.6-34b --prompt-mode visibility

python tools/rate_scripts/llava/eval_image_quality.py --mode cap-one \
--prompt-mode visibility --methods msbdn_semiv10_vlm12_round1
"""
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import requests
from PIL import Image
from io import BytesIO
import re

import os
import os.path as osp
import argparse
import socket
from datetime import datetime
import json
import pandas as pd

from tqdm import tqdm


ROOT_FOLS = {
    'pseudo-labels': [
        'data/RESIDE/URHI',
        'data/All-Weather/Serp/rain_filter_v1104',
        'data/All-Weather/Serp/snow_filter_v1104',
    ],
    'results-test': [
        'data/RESIDE/RTTS/JPEGImages',
        'data/All-Weather/RealRain_v1025',
        'data/All-Weather/Snow100K/realistic',
    ],
}

DST_FOLS = {
    'pseudo-labels': [
        'haze_train', 'rain_train', 'snow_train'
    ],
    'results-test': [
        'haze_test', 'rain_test', 'snow_test'
    ],
    'cap-one': [
        'haze_train', 'rain_train', 'snow_train'
    ],
    'cap-all': [
        'haze_train', 'rain_train', 'snow_train'
    ],
}

METHODS = [
    'input',
]

PREFIXES = {
    'pseudo-labels': './data/pseudo_labels',
    'results-test': './data/results_test',
    'cap-one': './data/pseudo_labels',
    'cap-all': './data/pseudo_labels',
}

CAP_INFO_FILES = {
    'cap-one': './data/caption_semi/caption_llama2-13b_both_one_v1122.json',
    'cap-all': './data/caption_semi/caption_llama2-13b_both_v1130.json',
}


def get_prompt():
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # if args.conv_mode is not None and conv_mode != args.conv_mode:
    #     print(
    #         "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
    #             conv_mode, args.conv_mode, args.conv_mode
    #         )
    #     )
    # else:
    #     args.conv_mode = conv_mode

    print(f"\nprompt: {qs}\n")

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt


def iqa_metric(images, prompt):
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_logits = []
        for image_tensor in images_tensor:
            logits = model(input_ids,
                           images=image_tensor.unsqueeze(0),
                           image_sizes=image_sizes,
                           )["logits"][:, -1]
            output_logits.append(logits)
        output_logits = torch.cat(output_logits, dim=0)

    return output_logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--prompt-mode", type=str, required=True)
    parser.add_argument("--methods", type=str, default=None)
    args = parser.parse_args()

    disable_torch_init()

    device = torch.device(f"cuda:0")

    # create metric with default setting
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        load_in_8bit=True,
        device=device
    )

    if args.prompt_mode == "visibility":
        # args.query = "Rate the visibility against rain/haze/snow."
        args.query = "Rate the visibility of the image against rain, fog, and snow. Please answer with excellent, good, fair, poor, or bad."
    elif args.prompt_mode == "quality":
        # args.query = "Rate the quality of the image."
        args.query = "Rate the quality of the image. Please answer with excellent, good, fair, poor, or bad."
    else:
        raise NotImplementedError
    prompt = get_prompt()

    toks = ["good", "poor", "high", "low", "excellent", "bad",
            "fine", "moderate", "decent", "average", "medium", "acceptable", "fair"]
    print(toks)
    ids_ = [id_[1] if 'v1.6-34b' not in args.model_path else id_[0]
            for id_ in tokenizer(toks)["input_ids"]]
    print(ids_)
    preferential_toks = ["excellent", "good", "fair", "poor", "bad"]
    print(preferential_toks)
    preferential_ids_ = [id_[1] if 'v1.6-34b' not in args.model_path else id_[0]
                         for id_ in tokenizer(preferential_toks)["input_ids"]]
    print(preferential_ids_)
    weight_tensor = torch.Tensor([5., 4., 3., 2., 1.]).float().to(model.device)
    print(weight_tensor)

    args.metric_model = f"{args.model_path.split('/')[-1]}-{args.prompt_mode}"
    print(f"\nmetric_model: {args.metric_model}")

    if 'cap' in args.mode:
        cap_info_file = CAP_INFO_FILES[args.mode]
        with open(cap_info_file, 'r') as f:
            cap_info = json.load(f)
    else:
        root_fols = ROOT_FOLS[args.mode]
        home_dir = '/home/jqxu'
        root_fols = [osp.join(home_dir, x) for x in root_fols]
        print(f"\nHome: {home_dir}")
    dst_fols = DST_FOLS[args.mode]

    prefix = PREFIXES[args.mode]

    if args.methods is None:
        methods = METHODS
    else:
        methods = args.methods.split(',')
    print(f"\nTesting: {methods}")

    for met in methods[:]:
        print(f"\nMethod: {met}\n")

        if 'cap' in args.mode:
            pbar = tqdm(cap_info)
            files, boxes, scores = [], [], []
            logits = {}

            i = 0

            for item in pbar:
                file = osp.split(item['image_file'])[-1]
                files.append(file)

                dst_img_path = osp.join(prefix, met, dst_fols[i], file)
                if not osp.isfile(dst_img_path):
                    i += 1
                    dst_img_path = osp.join(prefix, met, dst_fols[i], file)
                assert osp.isfile(dst_img_path), f"missing: {dst_img_path}"

                img = Image.open(dst_img_path)
                img = img.convert('RGB')

                box_ltrb = item['box_ltrb']
                img = img.crop(box_ltrb)
                boxes.append(box_ltrb)

                output_logits = iqa_metric([img], prompt)

                for tok, id_ in zip(toks, ids_):
                    if tok not in logits:
                        logits[tok] = []
                    logits[tok].append(output_logits[0, id_].item())

                score = output_logits[:, preferential_ids_]
                score = torch.softmax(score, -1).float() @ weight_tensor
                score = score.item()

                pbar.set_description(f"{dst_img_path[len(prefix) + 1:]}: {score:.4f}")

                scores.append(score)

            save_fol = osp.join('/'.join(osp.split(prefix)[:-1]), f"ratings_{args.mode}_" + osp.split(prefix)[-1], met)
            os.makedirs(save_fol, exist_ok=True)
            csv_path = osp.join(save_fol, f"scores-{args.metric_model}.csv")
            now = datetime.now()  # current date and time
            date_time = now.strftime("%Y/%m/%d, %H:%M:%S")
            print(f"[{date_time}] Saving ratings at: {csv_path}\n")

            if osp.isfile(csv_path):
                # TODO
                pass
            data = {'name': files, 'boxes': boxes}
            for k, v in logits.items():
                data[k] = v
            data[args.metric_model] = scores

            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
        else:
            for i in range(len(root_fols)):
                files = os.listdir(root_fols[i])
                files_dst = os.listdir(osp.join(prefix, met, dst_fols[i]))
                if len(files) != len(files_dst):
                    print(f"\nWarning!!!\n{root_fols[i]} [{len(files)}] vs."
                          f"{osp.join(prefix, met, dst_fols[i])} [{len(files_dst)}]")
                # assert len(files) == len(files_dst)  # not true for valid set
                files_dst.sort()
                pbar = tqdm(files_dst)
                scores = []
                logits = {}

                for file in pbar:
                    dst_img_path = osp.join(prefix, met, dst_fols[i], file)

                    assert osp.isfile(dst_img_path), f"missing: {dst_img_path}"

                    img = Image.open(dst_img_path)
                    img = img.convert('RGB')

                    output_logits = iqa_metric([img], prompt)

                    for tok, id_ in zip(toks, ids_):
                        if tok not in logits:
                            logits[tok] = []
                        logits[tok].append(output_logits[0, id_].item())

                    score = output_logits[:, preferential_ids_]
                    score = torch.softmax(score, -1).float() @ weight_tensor
                    score = score.item()

                    pbar.set_description(f"{dst_img_path[len(prefix) + 1:]}: {score:.4f}")

                    scores.append(score)

                save_fol = osp.join('/'.join(osp.split(prefix)[:-1]), 'ratings_' + osp.split(prefix)[-1], met)
                os.makedirs(save_fol, exist_ok=True)
                csv_path = osp.join(save_fol, f"scores-{args.metric_model}_{dst_fols[i]}.csv")
                now = datetime.now()  # current date and time
                date_time = now.strftime("%Y/%m/%d, %H:%M:%S")
                print(f"[{date_time}] Saving ratings at: {csv_path}")

                if osp.isfile(csv_path):
                    # TODO
                    pass
                data = {'name': files_dst}
                for k, v in logits.items():
                    data[k] = v
                data[args.metric_model] = scores

                df = pd.DataFrame(data)
                df.to_csv(csv_path, index=False)
