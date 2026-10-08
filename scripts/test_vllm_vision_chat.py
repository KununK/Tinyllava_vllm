from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from vllm import LLM, SamplingParams


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATHS = {
    "qwen2-0.5b": ROOT / "vllm_models/tinyllava-qwen2-0.5b-llava-vllm",
    "qwen2.5-3b": ROOT / "vllm_models/tinyllava-qwen2.5-3b-llava-vllm",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODEL_PATHS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument(
        "--image-url",
        default="http://images.cocodataset.org/test-stuff2017/000000000001.jpg",
    )
    parser.add_argument("--prompt", default="What are these?")
    args = parser.parse_args()

    gpu_memory_utilization = args.gpu_memory_utilization
    if gpu_memory_utilization is None:
        gpu_memory_utilization = 0.25 if args.model == "qwen2-0.5b" else 0.30

    image = Image.open(
        BytesIO(requests.get(args.image_url, timeout=20).content)
    ).convert("RGB")
    llm = LLM(
        model=str(MODEL_PATHS[args.model]),
        trust_remote_code=False,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )
    outputs = llm.generate(
        {
            "prompt": f"USER: <image>\n{args.prompt} ASSISTANT:",
            "multi_modal_data": {"image": image},
        },
        SamplingParams(max_tokens=80, temperature=0.0),
    )
    print(outputs[0].outputs[0].text.strip())


if __name__ == "__main__":
    main()
