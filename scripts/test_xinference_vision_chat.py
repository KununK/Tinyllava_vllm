from __future__ import annotations

import argparse
import base64
import os

import requests


MODEL_UIDS = {
    "qwen2-0.5b": "tinyllava-qwen2-0.5b-vllm",
    "qwen2.5-3b": "tinyllava-qwen2.5-3b-vllm",
}

os.environ["NO_PROXY"] = ",".join(
    part
    for part in [os.environ.get("NO_PROXY"), "127.0.0.1", "localhost"]
    if part
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODEL_UIDS)
    parser.add_argument("--endpoint", default=os.environ.get("XINFERENCE_ENDPOINT", "http://127.0.0.1:10003"))
    parser.add_argument("--image-url", default="http://images.cocodataset.org/test-stuff2017/000000000001.jpg")
    parser.add_argument("--prompt", default="What are these?")
    args = parser.parse_args()

    print(f"downloading image: {args.image_url}", flush=True)
    image_response = requests.get(args.image_url, timeout=30)
    image_response.raise_for_status()
    image_uri = "data:image/jpeg;base64," + base64.b64encode(image_response.content).decode("ascii")
    print(f"posting to: {args.endpoint}/v1/chat/completions", flush=True)

    payload = {
        "model": MODEL_UIDS[args.model],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "max_tokens": 128,
        "temperature": 0,
    }
    response = requests.post(f"{args.endpoint}/v1/chat/completions", json=payload, timeout=120)
    print(response.status_code)
    print(response.text)
    response.raise_for_status()


if __name__ == "__main__":
    main()
