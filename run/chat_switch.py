#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "run/models.json"


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_model_path(model: dict[str, Any]) -> str:
    path = str(model["model_path"])
    if model.get("source") == "builtin" and not path.startswith(("/", "./", "../")):
        return path
    return str(repo_path(path))


def model_accepts_image(model: dict[str, Any]) -> bool:
    limit_mm = model.get("limit_mm_per_prompt", {})
    return isinstance(limit_mm, dict) and int(limit_mm.get("image", 0) or 0) > 0


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def enabled_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [model for model in config["models"] if model.get("enabled", True)]


def model_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for model in enabled_models(config):
        result[model["alias"]] = model
        result[model["model_uid"]] = model
    return result


def endpoint(config: dict[str, Any]) -> str:
    server = config.get("server", {})
    return str(server.get("endpoint") or f"http://127.0.0.1:{server.get('port', 10003)}").rstrip("/")


def image_to_data_uri(value: str) -> str:
    if value.startswith("data:"):
        return value
    if value.startswith("http://") or value.startswith("https://"):
        response = requests.get(value, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
        return "data:" + content_type + ";base64," + base64.b64encode(response.content).decode("ascii")

    path = Path(value).expanduser()
    data = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")


def build_payload(model: dict[str, Any], prompt: str, image_uri: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    if image_uri and not model_accepts_image(model):
        raise ValueError(f"Model {model['alias']} has no image input enabled; clear image with /image clear.")

    if not image_uri:
        content: str | list[dict[str, Any]] = prompt
    else:
        content = [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": prompt},
        ]

    generation = dict(model.get("generation", {}))
    generation.update(overrides)
    payload = {
        "model": model["model_uid"],
        "messages": [{"role": "user", "content": content}],
    }
    for key in ("temperature", "top_p", "max_tokens"):
        if key in generation:
            payload[key] = generation[key]
    return payload


def post_chat(base_url: str, payload: dict[str, Any]) -> str:
    response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=180)
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code} {response.text}")
    data = response.json()
    return data["choices"][0]["message"]["content"]


def print_help() -> None:
    print(
        "Commands:\n"
        "  /models                  list models\n"
        "  /model <alias|uid>        switch model\n"
        "  /image <path|url>         set image for following turns\n"
        "  /image clear              clear image\n"
        "  /set temperature <value>  override temperature\n"
        "  /set top_p <value>        override top_p\n"
        "  /set max_tokens <value>   override max_tokens\n"
        "  /params                   show active params\n"
        "  /clear                    clear request overrides and image\n"
        "  /help                     show help\n"
        "  /exit                     quit"
    )


def interactive(config: dict[str, Any], current: dict[str, Any], base_url: str) -> None:
    models = model_map(config)
    image_uri: str | None = None
    image_label: str | None = None
    overrides: dict[str, Any] = {}

    print(f"Endpoint: {base_url}")
    print(f"Current model: {current['alias']} ({current['model_uid']})")
    print_help()

    while True:
        try:
            line = input(f"\n{current['alias']}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue

        if line in {"/exit", "/quit"}:
            return
        if line == "/help":
            print_help()
            continue
        if line == "/models":
            for model in enabled_models(config):
                source = model.get("source", "local")
                vision = "vision" if model_accepts_image(model) else "text"
                print(f"  {model['alias']}: source={source} mode={vision} uid={model['model_uid']} gpu={model['gpu_idx']} path={display_model_path(model)}")
            continue
        if line.startswith("/model "):
            name = line.split(maxsplit=1)[1]
            if name not in models:
                print(f"Unknown model: {name}")
                continue
            current = models[name]
            print(f"Switched to {current['alias']} ({current['model_uid']})")
            continue
        if line.startswith("/image "):
            value = line.split(maxsplit=1)[1].strip()
            if value == "clear":
                image_uri = None
                image_label = None
                print("Image cleared")
                continue
            try:
                image_uri = image_to_data_uri(value)
                image_label = value
                print(f"Image set: {value}")
            except Exception as exc:
                print(f"Failed to load image: {exc}")
            continue
        if line.startswith("/set "):
            parts = line.split(maxsplit=2)
            if len(parts) != 3 or parts[1] not in {"temperature", "top_p", "max_tokens"}:
                print("Usage: /set {temperature|top_p|max_tokens} <value>")
                continue
            key = parts[1]
            overrides[key] = int(parts[2]) if key == "max_tokens" else float(parts[2])
            print(f"{key}={overrides[key]}")
            continue
        if line == "/params":
            generation = dict(current.get("generation", {}))
            generation.update(overrides)
            print(json.dumps({"model": current["model_uid"], "image": image_label, **generation}, ensure_ascii=False, indent=2))
            continue
        if line == "/clear":
            image_uri = None
            image_label = None
            overrides.clear()
            print("Cleared image and request overrides")
            continue

        payload = build_payload(current, line, image_uri, overrides)
        try:
            print(post_chat(base_url, payload).strip())
        except Exception as exc:
            print(f"ERROR: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    os.environ["NO_PROXY"] = ",".join(
        part for part in [os.environ.get("NO_PROXY"), "127.0.0.1", "localhost"] if part
    )
    config = load_config(args.config)
    models = model_map(config)
    current = models.get(args.model or "", enabled_models(config)[0])
    base_url = (args.endpoint or endpoint(config)).rstrip("/")

    if args.prompt:
        image_uri = image_to_data_uri(args.image) if args.image else None
        payload = build_payload(current, args.prompt, image_uri, {})
        print(post_chat(base_url, payload).strip())
        return

    interactive(config, current, base_url)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
