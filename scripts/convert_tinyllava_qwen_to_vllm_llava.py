from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, LlavaProcessor, SiglipImageProcessor


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATHS = {
    "qwen2-0.5b": ROOT.parent / "TinyLLaVA-Qwen2-0.5B-SigLIP",
    "qwen2.5-3b": ROOT.parent / "TinyLLaVA-Qwen2.5-3B-SigLIP",
}

SKIP_SUFFIXES = {
    ".safetensors",
    ".bin",
}
SKIP_NAMES = {
    "model.safetensors.index.json",
    "trainer_state.json",
    "training_args.bin",
}


def rename_key(key: str) -> str:
    replacements = {
        "connector._connector.0.": "multi_modal_projector.linear_1.",
        "connector._connector.2.": "multi_modal_projector.linear_2.",
        "vision_tower._vision_tower.": "vision_tower.",
    }
    for old, new in replacements.items():
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def copy_metadata_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            continue
        if item.name in SKIP_NAMES or item.suffix in SKIP_SUFFIXES:
            continue
        shutil.copy2(item, dst / item.name)


def save_tokenizer_and_processor(src: Path, dst: Path, config: dict) -> int:
    tokenizer = AutoTokenizer.from_pretrained(src, use_fast=False, trust_remote_code=True)
    if tokenizer.convert_tokens_to_ids("<image>") == tokenizer.unk_token_id:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    tokenizer.save_pretrained(dst)

    vision_config = config["vision_config"]
    feature_select_strategy = config.get("vision_feature_select_strategy", "default")
    if feature_select_strategy == "patch":
        feature_select_strategy = "default"
    image_processor = SiglipImageProcessor(
        do_resize=True,
        size={
            "height": vision_config["image_size"],
            "width": vision_config["image_size"],
        },
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    processor = LlavaProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=vision_config["patch_size"],
        vision_feature_select_strategy=feature_select_strategy,
        image_token="<image>",
        num_additional_image_tokens=0,
    )
    processor.save_pretrained(dst)
    return int(image_token_id)


def write_config(src: Path, dst: Path, image_token_id: int) -> None:
    with (src / "config.json").open() as f:
        config = json.load(f)

    config["architectures"] = ["LlavaForConditionalGeneration"]
    config["model_type"] = "llava"
    config["image_token_index"] = image_token_id
    config["projector_hidden_act"] = "gelu"
    config["multimodal_projector_bias"] = True
    if config.get("vision_feature_select_strategy") == "patch":
        config["vision_feature_select_strategy"] = "default"
    config["image_seq_length"] = (
        config["vision_config"]["image_size"] // config["vision_config"]["patch_size"]
    ) ** 2

    config.pop("auto_map", None)
    with (dst / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def convert_safetensors_file(src_file: Path, dst_file: Path) -> list[str]:
    tensors = {}
    with safe_open(src_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[rename_key(key)] = f.get_tensor(key)
    save_file(tensors, dst_file)
    return sorted(tensors)


def convert_weights(src: Path, dst: Path) -> None:
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            index = json.load(f)

        new_weight_map = {}
        for shard_name in sorted(set(index["weight_map"].values())):
            src_shard = src / shard_name
            dst_shard = dst / shard_name
            converted_keys = convert_safetensors_file(src_shard, dst_shard)
            for key in converted_keys:
                new_weight_map[key] = shard_name

        new_index = {
            "metadata": index.get("metadata", {}),
            "weight_map": dict(sorted(new_weight_map.items())),
        }
        with (dst / "model.safetensors.index.json").open("w") as f:
            json.dump(new_index, f, indent=2)
            f.write("\n")
        return

    convert_safetensors_file(src / "model.safetensors", dst / "model.safetensors")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODEL_PATHS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "vllm_models",
    )
    args = parser.parse_args()

    src = MODEL_PATHS[args.model]
    dst = args.output_root / f"tinyllava-{args.model}-llava-vllm"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    with (src / "config.json").open() as f:
        source_config = json.load(f)

    copy_metadata_files(src, dst)
    image_token_id = save_tokenizer_and_processor(src, dst, source_config)
    write_config(src, dst, image_token_id)
    convert_weights(src, dst)

    print(dst)
    print(f"image_token_id={image_token_id}")


if __name__ == "__main__":
    main()
