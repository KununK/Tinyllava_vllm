# TinyLLaVA vLLM on Xinference

This repository contains scripts for serving local TinyLLaVA Qwen checkpoints and native Hugging Face text models with Xinference and the vLLM engine.

Raw TinyLLaVA checkpoints use custom Hugging Face remote code with `model_type: tinyllava`. vLLM does not load that layout directly in this environment, so the workflow converts the Qwen TinyLLaVA checkpoints into a vLLM-compatible `LlavaForConditionalGeneration` layout and registers matching Xinference model specs.

## Supported Models

Configured in `run/models.json`:

- `tinyllava-qwen2-0.5b-vllm`: TinyLLaVA Qwen2 0.5B vision-language model
- `tinyllava-qwen2.5-3b-vllm`: TinyLLaVA Qwen2.5 3B vision-language model
- `qwen3-8b-vllm`: native text-only `Qwen/Qwen3-8B`

Expected source checkpoints, relative to this repository:

- `../TinyLLaVA-Qwen2-0.5B-SigLIP`
- `../TinyLLaVA-Qwen2.5-3B-SigLIP`

Generated vLLM checkpoints:

- `vllm_models/tinyllava-qwen2-0.5b-llava-vllm`
- `vllm_models/tinyllava-qwen2.5-3b-llava-vllm`

Model weights and converted checkpoints are not tracked by Git. Native Hugging Face models such as `Qwen/Qwen3-8B` are resolved by Xinference/vLLM from the Hugging Face cache or downloaded by the runtime if network access is available.

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

- `kind`: `tinyllava` for converted TinyLLaVA VLMs, `text` for native text-only HF models
- model paths
- GPU placement
- `context_length`
- `gpu_memory_utilization`
- default generation parameters such as `temperature`, `top_p`, and `max_tokens`

For converted TinyLLaVA models, `model_path` must point to a local vLLM checkpoint directory. For native text models that Xinference already knows, such as `Qwen/Qwen3-8B`, a non-local Hugging Face repo id in `model_path` is treated as metadata and the launcher omits `--model-path`, letting Xinference resolve the builtin model source.

Deploy all enabled models:

```bash
.venv/bin/python run/deploy_multi_models.py
```

By default this starts Xinference at:

```text
http://127.0.0.1:10003
```

All models share the same OpenAI-compatible endpoint:

```text
POST http://127.0.0.1:10003/v1/chat/completions
```

Select the model with the request body's `model` field:

- `tinyllava-qwen2-0.5b-vllm`
- `tinyllava-qwen2.5-3b-vllm`
- `qwen3-8b-vllm`

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

For text-only models, clear any image context before switching:

```text
/image clear
/model qwen3-8b
你好
```

One-shot call:

```bash
.venv/bin/python run/chat_switch.py \
  --model qwen2-0.5b \
  --image http://images.cocodataset.org/test-stuff2017/000000000001.jpg \
  --prompt "What are these?"
```

## Conversation State

The OpenAI-compatible API is stateless. vLLM and Xinference do not remember prior turns for a model just because the same client keeps sending requests.

To keep context, the client must send the conversation history in every request:

```json
{
  "model": "qwen3-8b-vllm",
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"},
    {"role": "user", "content": "继续解释刚才的问题"}
  ],
  "max_tokens": 512
}
```

`run/chat_switch.py` is a lightweight smoke-test shell and currently sends each user input as a fresh single-turn request. It is expected to have no memory unless you extend it to append previous `user` and `assistant` messages to the next payload.

## Token And Memory Parameters

`generation.max_tokens` in `run/models.json` controls the maximum number of new tokens generated by one request. Increasing it allows longer answers, but the model may still stop earlier when it reaches EOS or a stop condition.

`context_length` controls the maximum total sequence length accepted by vLLM for the model. It must cover prompt tokens plus generated tokens:

```text
prompt/history/image tokens + max_tokens <= context_length
```

You do not need to increase `context_length` just because you increase `max_tokens`, as long as the total still fits. For example, `context_length=4096` and `max_tokens=512` leaves roughly 3584 tokens for prompt and history.

Memory impact:

- Increasing `max_tokens` mostly increases per-request KV cache usage while that request is running.
- Increasing `context_length` increases the KV cache budget vLLM plans for, which can reduce concurrency or require more GPU memory.
- Increasing `gpu_memory_utilization` lets vLLM reserve more GPU memory for KV cache, usually improving capacity/concurrency but leaving less room for other processes.
- Model weights dominate base memory. KV cache grows with sequence length, batch/concurrency, number of layers, hidden size, and dtype.

For longer conversations, first raise `max_tokens` only. Raise `context_length` only when prompts plus history plus output exceed the current limit.

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
