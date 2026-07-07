# Multi-model Xinference + vLLM Runner

Configuration:

- `models.json`: model paths, GPU placement, vLLM launch options, and default generation params.
- TinyLLaVA `model_path` values are local converted checkpoint directories. Text-only builtin HF models can keep repo ids such as `Qwen/Qwen3-8B`; the launcher will not pass those repo ids as local `--model-path` values.

Deploy all enabled models:

```bash
cd Tinyllava_vllm
.venv/bin/python run/deploy_multi_models.py --config run/models.json
```

Restart already loaded model actors:

```bash
.venv/bin/python run/deploy_multi_models.py --restart-models
```

Terminate the two model actors but keep Xinference running:

```bash
.venv/bin/python run/deploy_multi_models.py --terminate-models
```

Stop the Xinference service started by this runner:

```bash
.venv/bin/python run/deploy_multi_models.py --stop-server
```

Interactive shell:

```bash
.venv/bin/python run/chat_switch.py
```

One-shot test:

```bash
.venv/bin/python run/chat_switch.py \
  --model qwen2-0.5b \
  --image http://images.cocodataset.org/test-stuff2017/000000000001.jpg \
  --prompt "What are these?"
```

Text-only Qwen3 one-shot test:

```bash
.venv/bin/python run/chat_switch.py \
  --model qwen3-8b \
  --prompt "用一句话介绍一下你自己。"
```

Context:

- The API is stateless. The server does not remember old turns automatically.
- `run/chat_switch.py` is a smoke-test shell and sends one fresh request per input.
- To make a real multi-turn chat client, keep a local `messages` list and send the full history every time.

Token settings:

- `generation.max_tokens` is the max number of newly generated tokens.
- `context_length` is the max total length: prompt/history/image tokens plus generated tokens.
- Increase `max_tokens` for longer answers. Increase `context_length` only if the total request length does not fit.
- Larger `context_length` and more concurrent/long requests consume more KV-cache GPU memory.

All OpenAI-compatible APIs share one Xinference endpoint and differ by `model`:

- `POST http://127.0.0.1:10003/v1/chat/completions`, `model=tinyllava-qwen2-0.5b-vllm`
- `POST http://127.0.0.1:10003/v1/chat/completions`, `model=tinyllava-qwen2.5-3b-vllm`
- `POST http://127.0.0.1:10003/v1/chat/completions`, `model=qwen3-8b-vllm`
