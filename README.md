# TinyLLaVA vLLM on Xinference

This repository contains scripts for serving local TinyLLaVA Qwen checkpoints with Xinference and the vLLM engine.

Raw TinyLLaVA checkpoints use custom Hugging Face remote code with `model_type: tinyllava`. vLLM does not load that layout directly in this environment, so the workflow converts the Qwen TinyLLaVA checkpoints into a vLLM-compatible `LlavaForConditionalGeneration` layout and registers matching Xinference model specs.

## Supported Models

Expected source checkpoints, relative to this repository:

- `../TinyLLaVA-Qwen2-0.5B-SigLIP`
- `../TinyLLaVA-Qwen2.5-3B-SigLIP`

Generated vLLM checkpoints:

- `vllm_models/tinyllava-qwen2-0.5b-llava-vllm`
- `vllm_models/tinyllava-qwen2.5-3b-llava-vllm`

Model weights and converted checkpoints are not tracked by Git.

## Requirements

- Linux
- NVIDIA GPU with CUDA available
- `uv`
- Python 3.11
- Local TinyLLaVA source checkpoint directories listed above

The environment used during development was:

- `xinference==2.9.0`
- `vllm==0.21.0`
- `transformers==5.9.0`
- `torch==2.11.0`

## Setup

```bash
cd Tinyllava_vllm
uv --cache-dir .uv-cache sync
.venv/bin/python scripts/patch_xinference_tinyllava_qwen.py
```

Check the environment:

```bash
.venv/bin/python scripts/check_env.py
```

## Convert Checkpoints

```bash
.venv/bin/python scripts/convert_tinyllava_qwen_to_vllm_llava.py qwen2-0.5b
.venv/bin/python scripts/convert_tinyllava_qwen_to_vllm_llava.py qwen2.5-3b
```

Optional direct vLLM smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/test_vllm_vision_chat.py qwen2-0.5b
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/test_vllm_vision_chat.py qwen2.5-3b
```

## Single-model Xinference Test

Start Xinference:

```bash
./scripts/start_xinference.sh
```

In another shell:

```bash
export XINFERENCE_ENDPOINT=http://127.0.0.1:10003
./scripts/register_xinference_tinyllava_qwen.sh
./scripts/launch_xinference_vllm_single.sh qwen2-0.5b 0
.venv/bin/python scripts/test_xinference_vision_chat.py qwen2-0.5b
```

## Multi-model Deployment

Edit `run/models.json` to set:

- model paths
- GPU placement
- `context_length`
- `gpu_memory_utilization`
- default generation parameters such as `temperature`, `top_p`, and `max_tokens`

Deploy both models:

```bash
.venv/bin/python run/deploy_multi_models.py
```

By default this starts Xinference at:

```text
http://127.0.0.1:10003
```

Both models share the same OpenAI-compatible endpoint:

```text
POST http://127.0.0.1:10003/v1/chat/completions
```

Select the model with the request body's `model` field:

- `tinyllava-qwen2-0.5b-vllm`
- `tinyllava-qwen2.5-3b-vllm`

List available models:

```bash
curl http://127.0.0.1:10003/v1/models
```

## Interactive Test Shell

```bash
.venv/bin/python run/chat_switch.py
```

Useful commands inside the shell:

```text
/models
/model qwen2.5-3b
/image http://images.cocodataset.org/test-stuff2017/000000000001.jpg
/set temperature 0.2
/params
/exit
```

One-shot call:

```bash
.venv/bin/python run/chat_switch.py \
  --model qwen2-0.5b \
  --image http://images.cocodataset.org/test-stuff2017/000000000001.jpg \
  --prompt "What are these?"
```

## Stop Services

Terminate model actors but keep Xinference running:

```bash
.venv/bin/python run/deploy_multi_models.py --terminate-models
```

Terminate models and stop the Xinference process started by this runner:

```bash
.venv/bin/python run/deploy_multi_models.py --stop-server
```

## Repository Layout

- `scripts/`: conversion, patching, single-model launch, and smoke tests
- `xinference_models/`: custom Xinference model registrations
- `run/models.json`: multi-model deployment configuration
- `run/deploy_multi_models.py`: multi-model deployment runner
- `run/chat_switch.py`: interactive API test shell

Generated directories such as `.venv/`, `.uv-cache/`, `.xinference_multi/`, `run/logs/`, and `vllm_models/` are intentionally ignored.
