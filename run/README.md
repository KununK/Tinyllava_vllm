# Multi-model Xinference + vLLM Runner

Configuration:

- `models.json`: model paths, GPU placement, vLLM launch options, and default generation params.

Deploy both models:

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

The two OpenAI-compatible APIs share one Xinference endpoint and differ by `model`:

- `POST http://127.0.0.1:10003/v1/chat/completions`, `model=tinyllava-qwen2-0.5b-vllm`
- `POST http://127.0.0.1:10003/v1/chat/completions`, `model=tinyllava-qwen2.5-3b-vllm`
