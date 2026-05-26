from __future__ import annotations

import importlib.metadata as metadata
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from transformers import AutoTokenizer

SOURCE_MODEL_DIRS = {
    "qwen2-0.5b": ROOT.parent / "TinyLLaVA-Qwen2-0.5B-SigLIP",
    "qwen2.5-3b": ROOT.parent / "TinyLLaVA-Qwen2.5-3B-SigLIP",
}

CONVERTED_MODEL_DIRS = {
    "qwen2-0.5b": ROOT / "vllm_models/tinyllava-qwen2-0.5b-llava-vllm",
    "qwen2.5-3b": ROOT / "vllm_models/tinyllava-qwen2.5-3b-llava-vllm",
}


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    print("Environment")
    for package in ("xinference", "vllm", "torch", "transformers", "accelerate"):
        print(f"  {package}: {version(package)}")

    print("\nSource model dirs")
    for name, model_dir in SOURCE_MODEL_DIRS.items():
        status = "OK" if (model_dir / "config.json").exists() else "MISSING"
        print(f"  {name}: {status} {model_dir}")

    print("\nConverted vLLM model dirs")
    for name, model_dir in CONVERTED_MODEL_DIRS.items():
        if not (model_dir / "config.json").exists():
            print(f"  {name}: MISSING {model_dir}")
            continue

        with (model_dir / "config.json").open() as config_file:
            config = json.load(config_file)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        print(
            "  "
            f"{name}: model_type={config.get('model_type')}, "
            f"arch={','.join(config.get('architectures') or [])}, "
            f"image_token_id={image_token_id}, "
            f"path={model_dir}"
        )

    import xinference

    xinference_root = Path(xinference.__file__).resolve().parent
    patched = all(
        marker in (xinference_root / relative).read_text()
        for relative, marker in {
            "model/llm/vllm/core.py": "TinyLLaVA Qwen vLLM adapter",
            "core/model.py": "tinyllava_chat",
            "api/restful_api.py": "model.tinyllava_chat",
        }.items()
    )
    print(f"\nXinference TinyLLaVA patch: {'OK' if patched else 'MISSING'}")


if __name__ == "__main__":
    main()
