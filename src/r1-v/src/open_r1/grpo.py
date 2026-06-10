# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time
from datetime import datetime


def _patch_av_averror_for_torchvision() -> None:
    """
    Torchvision's `io/video.py` uses `except av.AVError` on the `av` module object.
    Some PyAV builds only define the class as `av.error.AVError` — bind everywhere.
    Safe to call repeatedly (e.g. before `trainer.train()`).
    """
    try:
        import importlib
        import sys

        import av
        import av.error

        err_cls = getattr(av.error, "AVError", getattr(av.error, "FFmpegError", Exception))
        av.AVError = err_cls  # type: ignore[attr-defined]
        m = sys.modules.get("av")
        if m is not None and getattr(m, "AVError", None) is not err_cls:
            m.AVError = err_cls  # type: ignore[attr-defined]
        try:
            tvv = importlib.import_module("torchvision.io.video")
            if getattr(tvv, "av", None) is not None:
                tvv.av.AVError = err_cls  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


_patch_av_averror_for_torchvision()
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )



def accuracy_reward(completions, solution, **kwargs):
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        if num_str is None or not str(num_str).strip():
            return None
        try:
            num_str = str(num_str).replace(",", "")
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    

    question_type = kwargs['problem_type'][0]
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
    
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                else:
                    rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                    rel_diff = min(1.0, max(0.0, rel_diff))
                    reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def _resolve_local_model_dir(model_id: str) -> str:
    """
    If ``model_id`` is clearly a filesystem path, require an existing HF checkpoint directory.
    Otherwise ``from_pretrained`` may treat a missing path like a Hub id and raise HFValidationError.
    """
    mid = model_id.strip()
    expanded = os.path.abspath(os.path.expanduser(mid))
    likely_fs = (
        os.path.isabs(mid)
        or mid.startswith(("./", "../", "~/"))
        or (len(mid) > 1 and mid[1] == ":")
    )
    if not likely_fs:
        return model_id
    if not os.path.isdir(expanded):
        raise FileNotFoundError(
            "[Video-R1] --model_name_or_path looks like a local path but is not a directory:\n"
            f"  input: {mid!r}\n"
            f"  resolved: {expanded!r}\n"
            "Fix the path, NFS mount, or symlink. If the folder is missing, transformers falls through to the Hub and "
            "you get huggingface_hub.errors.HFValidationError on the path string."
        )
    cfg = os.path.join(expanded, "config.json")
    if not os.path.isfile(cfg):
        raise FileNotFoundError(
            "[Video-R1] local model directory has no config.json (not a saved HF model folder):\n"
            f"  {expanded!r}"
        )
    return expanded


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    
    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    def make_conversation_image(example):
        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
        
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }
        
    def make_conversation_image_and_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            # example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                        ]
                }]
            }
        
        return msg

    
    # Parallel map speeds up startup a lot; flaky NFS can set VIDEO_R1_MAP_NUM_PROC=1.
    if os.environ.get("VIDEO_R1_MAP_NUM_PROC", "").strip():
        _map_workers = max(1, int(os.environ["VIDEO_R1_MAP_NUM_PROC"]))
    else:
        _cpu = os.cpu_count() or 8
        _map_workers = max(1, min(16, _cpu // 2))
    _map_batch = max(100, int(os.environ.get("VIDEO_R1_MAP_WRITER_BATCH", "1000")))
    print(
        f"[Video-R1] dataset.map num_proc={_map_workers} writer_batch_size={_map_batch} "
        f"(set VIDEO_R1_MAP_NUM_PROC / VIDEO_R1_MAP_WRITER_BATCH to tune; 2nd run uses HF cache)",
        flush=True,
    )
    try:
        if isinstance(dataset, DatasetDict):
            for _k, _ds in dataset.items():
                _n = getattr(_ds, "num_rows", len(_ds))
                print(f"[Video-R1] before map split={_k!r} rows={_n}", flush=True)
        else:
            _n = getattr(dataset, "num_rows", len(dataset))
            print(f"[Video-R1] before map rows={_n}", flush=True)
    except Exception:
        pass
    _map_t0 = time.perf_counter()
    dataset = dataset.map(
        make_conversation_image_and_video,
        desc="make_conversation_image_and_video",
        num_proc=_map_workers,
        load_from_cache_file=True,
        writer_batch_size=_map_batch,
    )
    print(f"[Video-R1] dataset.map finished in {time.perf_counter() - _map_t0:.1f}s", flush=True)

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    _lr = os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    if str(_lr) == "0":
        print(
            "[Video-R1] constructing trainer — [GRPO init] lines follow on local rank 0 by default "
            "(VIDEO_R1_INIT_PROGRESS=0 to silence, =all for every rank)",
            flush=True,
        )

    _model_path = (
        _resolve_local_model_dir(model_args.model_name_or_path)
        if isinstance(model_args.model_name_or_path, str)
        else model_args.model_name_or_path
    )

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=_model_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    _patch_av_averror_for_torchvision()
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
