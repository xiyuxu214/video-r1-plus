# -*- coding: utf-8 -*-
"""Offline benchmark driver for Video-R1 eval JSON; UTF-8 source only (ASCII in docstrings)."""
import os
import json
import re
import warnings
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse


_mm_kw_extra_warned: set[str] = set()


def _patch_vllm_qwen25_vl_processing_info() -> None:
    """Strip extra mm kwargs for vLLM 0.7.x + Qwen2.5-VL.

    vLLM calls get_hf_processor(**kwargs) / get_image_processor(**kwargs) with HF keys
    (e.g. do_sample_frames) but Qwen2_5_VLProcessingInfo only accepts min_pixels, max_pixels,
    fps. Drop unknown keys to avoid TypeError. No-op if vllm layout differs.
    """
    try:
        from vllm.model_executor.models import qwen2_5_vl as q25
    except Exception:
        return
    cls = getattr(q25, "Qwen2_5_VLProcessingInfo", None)
    if cls is None:
        return
    if getattr(cls, "_eval_bench_mm_kwargs_patch", False):
        return

    _KNOWN_DROP = frozenset(
        {
            "do_sample_frames",
            "num_frames",
            "video_fps",
            "second_per_grid_ts",
        }
    )

    def _sanitize(*, min_pixels=None, max_pixels=None, fps=2.0, **kwargs):
        if "fps" in kwargs and kwargs["fps"] is not None:
            fps = kwargs.pop("fps")
        if "min_pixels" in kwargs:
            min_pixels = kwargs.pop("min_pixels")
        if "max_pixels" in kwargs:
            max_pixels = kwargs.pop("max_pixels")
        for k in _KNOWN_DROP:
            kwargs.pop(k, None)
        if kwargs:
            key = ",".join(sorted(kwargs.keys()))
            if key not in _mm_kw_extra_warned:
                _mm_kw_extra_warned.add(key)
                warnings.warn(
                    "eval_bench: ignoring unsupported mm kwargs for Qwen2.5-VL / vLLM 0.7.x "
                    f"(once): {key}",
                    stacklevel=3,
                )
        return min_pixels, max_pixels, fps

    _orig_hf = cls.get_hf_processor
    _orig_img = cls.get_image_processor

    def get_hf_processor(self, *, min_pixels=None, max_pixels=None, fps=2.0, **kwargs):
        min_pixels, max_pixels, fps = _sanitize(
            min_pixels=min_pixels, max_pixels=max_pixels, fps=fps, **kwargs
        )
        return _orig_hf(self, min_pixels=min_pixels, max_pixels=max_pixels, fps=fps)

    def get_image_processor(self, *, min_pixels=None, max_pixels=None, fps=2.0, **kwargs):
        min_pixels, max_pixels, fps = _sanitize(
            min_pixels=min_pixels, max_pixels=max_pixels, fps=fps, **kwargs
        )
        return _orig_img(self, min_pixels=min_pixels, max_pixels=max_pixels, fps=fps)

    cls.get_hf_processor = get_hf_processor  # type: ignore[method-assign]
    cls.get_image_processor = get_image_processor  # type: ignore[method-assign]
    cls._eval_bench_mm_kwargs_patch = True


parser = argparse.ArgumentParser(description="Evaluation benchmark")
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Samples per vLLM batch (lower if host OOM-kills the job). Default: 64.",
)
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--file_name', type=str, required=True, help="Name of the file")
parser.add_argument(
    "--tensor_parallel_size",
    type=int,
    default=None,
    help="vLLM tensor parallel size; must divide the model's attention head count "
    "(e.g. Qwen2.5-VL-7B has 28). Default: torch.cuda.device_count().",
)
parser.add_argument(
    "--max_model_len",
    type=int,
    default=8192 * 2,
    help="vLLM max sequence length (KV cache grows with this). Lower if CUDA OOM during "
    "engine init (e.g. 8192 or 4096). Default: 16384.",
)
parser.add_argument(
    "--gpu_memory_utilization",
    type=float,
    default=0.8,
    help="Fraction of GPU memory vLLM may reserve. Lower if OOM or other jobs share "
    "the GPU (e.g. 0.55). Default: 0.8.",
)
parser.add_argument(
    "--disable_custom_all_reduce",
    action="store_true",
    help="Disable vLLM custom all-reduce (often fixes TP cudagraph / "
    "custom_all_reduce.cuh 'invalid argument' on some nodes).",
)
parser.add_argument(
    "--enforce_eager",
    action="store_true",
    help="Disable CUDA graph capture (slower; use if capture still fails after "
    "--disable_custom_all_reduce).",
)
args = parser.parse_args()

MODEL_PATH = args.model_path
file_name = args.file_name
BSZ = max(1, args.batch_size)

_tensor_parallel_size = (
    args.tensor_parallel_size
    if args.tensor_parallel_size is not None
    else torch.cuda.device_count()
)

_patch_vllm_qwen25_vl_processing_info()

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=_tensor_parallel_size,
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    limit_mm_per_prompt={"image": 1, "video": 1},
    disable_custom_all_reduce=args.disable_custom_all_reduce,
    enforce_eager=args.enforce_eager,
)


sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=1024,
    stop_token_ids=[],
)


processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer


def _resolve_media_path(cwd: str, path_field: str) -> str:
    """Join repo-relative media paths; JSON often uses ./Evaluation/... under src/r1-v/."""
    p = path_field.replace("\\", "/").strip()
    if p.startswith("./"):
        p = p[2:]
    return os.path.normpath(os.path.join(cwd, "src", "r1-v", p))


for dataset_name in ['mvbench','tempcompass','videomme','videommmu','vsibench','mmvu']:

    OUTPUT_PATH = f"./src/r1-v/eval_results/eval_{dataset_name}_{file_name}_greedy_output.json"
    PROMPT_PATH = f"./src/r1-v/Evaluation/eval_{dataset_name}.json"

    if not os.path.isfile(PROMPT_PATH):
        print(f"[skip] missing prompt file: {PROMPT_PATH}")
        continue
    
    if PROMPT_PATH.endswith('.jsonl'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif PROMPT_PATH.endswith('.json'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Input file must be .json or .jsonl")

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }


    messages = []
    for x in data:
        if x["problem_type"] == 'multiple choice':
            question = x['problem'] + "Options:\n"
            for op in x["options"]:
                question += op + "\n"
        else:
            question = x['problem']

        msg = [{
            "role": "user",
            "content": [
                {
                    "type": x['data_type'],
                    x['data_type']: _resolve_media_path(os.getcwd(), x['path']),
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x['problem_type']]
                }
            ]
        }]
        messages.append(msg)
        

    final_output = []
    start_idx = 0
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                final_output = existing.get("results", [])
                start_idx = len(final_output)
                print(f"Resuming from sample index {start_idx}")
        except Exception as e:
            print(f"Error reading existing output file: {e}")


    def extract_think(output_str):
        pattern = r'<think>\s*(.*?)\s*</think>'
        match = re.search(pattern, output_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            return None
        
    def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):

        if not torch.is_tensor(pred):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float32)
        
        epsilon = 1e-8
        rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
        
        thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
        
        conditions = rel_error < (1 - thresholds)  
        mra = conditions.float().mean()  
        return mra.item()


    def reward_fn(sample, model_output, question_type):
        try:
            output_ans = extract_answer(model_output)
            if output_ans == '':
                output_ans = model_output
            gt_ans = extract_answer(sample.get("solution", ""))
            if question_type == "multiple choice":
                return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    return 0.0
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                mra = mean_relative_accuracy(out_number, gt_number)
                return mra
            else:
                return 0.0
        except Exception as e:
            return 0.0

    for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
        batch_messages = messages[i:i + BSZ]

        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
            
            image_idx = 0
            video_idx = 0

            llm_inputs = []

            
            for idx, prompt in enumerate(prompts):
                mm_type = batch_messages[idx][0]['content'][0]['type']
                sample_mm_data = {}
                sample_video_kw = {}
                if mm_type == 'image':
                    sample_mm_data["image"] = image_inputs[image_idx]
                    image_idx += 1
                elif mm_type == 'video':
                    sample_mm_data["video"] = video_inputs[video_idx]
                    for key, value in (video_kwargs or {}).items():
                        # qwen_vl_utils may mix per-video lists with scalar flags; only index sequences.
                        if isinstance(value, (list, tuple)):
                            sample_video_kw[key] = value[video_idx]
                        else:
                            sample_video_kw[key] = value
                    # Newer vLLM / HF stacks may expect video_fps; qwen_vl_utils only returns fps.
                    if "fps" in sample_video_kw and "video_fps" not in sample_video_kw:
                        sample_video_kw["video_fps"] = sample_video_kw["fps"]
                    video_idx += 1
                        
                
                llm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                })
                

            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            batch_output_text = [out.outputs[0].text for out in outputs]
            
        except Exception as e:
            print('error:', data[i]['path'])
            print('Exception:', e)
            batch_output_text = ['<answer>error</answer>'] * BSZ
            

        for j, (sample, model_output) in enumerate(zip(data[i:i+BSZ], batch_output_text), start=i):
            think_chain = extract_think(model_output)
            final_ans = extract_answer(model_output)
            if final_ans == "":
                final_ans = model_output
            sample["output"] = model_output
            sample["prediction"] = final_ans
            q_type = sample.get("problem_type", "")
            sample["reward"] = reward_fn(sample, model_output, q_type)
            sample['correct'] = True if sample["reward"]==1.0 else False
            if think_chain:
                sample["process"] = f"<think>{think_chain}</think>"
            final_output.append(sample)
        

        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
        except Exception as e:
            print(f"Error writing to output file: {e}")

    # Recompute metrics from full results (fixes wrong 0.0 when this run had 0 batches but resumed file).
    acc_vals = [
        float(x["reward"])
        for x in final_output
        if x.get("problem_type") != "regression" and isinstance(x.get("reward"), (int, float))
    ]
    mra_vals = [
        float(x["reward"])
        for x in final_output
        if x.get("problem_type") == "regression" and isinstance(x.get("reward"), (int, float))
    ]
    final_acc = {"mean_acc": 0.0, "mean_mra": 0.0}
    if acc_vals:
        final_acc["mean_acc"] = torch.tensor(acc_vals, dtype=torch.float32).mean().item()
    if mra_vals:
        final_acc["mean_mra"] = torch.tensor(mra_vals, dtype=torch.float32).mean().item()

    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
        print(f"Final accuracy saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error writing final accuracy to output file: {e}")
    
    print(f"Results saved to {OUTPUT_PATH}")
