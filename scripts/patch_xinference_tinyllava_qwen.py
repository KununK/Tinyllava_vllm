from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


PATCH_MARKER = "TinyLLaVA Qwen vLLM adapter"


TINYLLAVA_BASE_METHODS = '''    @staticmethod
    def _tinyllava_load_image(url: str):
        import base64
        import re
        from io import BytesIO

        import requests
        from PIL import Image

        data_uri_pattern = re.compile(
            r"data:([a-zA-Z0-9]+/[a-zA-Z0-9-.+]+);base64,(.*)"
        )
        data_uri_match = data_uri_pattern.match(url)
        if data_uri_match:
            image_bytes = base64.b64decode(data_uri_match.group(2))
        elif url.startswith("http://") or url.startswith("https://"):
            image_bytes = requests.get(url, timeout=30).content
        else:
            return Image.open(url).convert("RGB")
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    def _tinyllava_build_inputs(self, messages: List[ChatCompletionMessage]):
        images = []
        prompt_parts = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role != "user":
                continue

            text_parts = []
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if "text" in item:
                        text_parts.append(item["text"])
                    elif "image_url" in item:
                        image_url = item["image_url"]
                        if isinstance(image_url, dict):
                            image_url = image_url.get("url")
                        if image_url:
                            images.append(self._tinyllava_load_image(str(image_url)))
                    elif "image" in item:
                        images.append(self._tinyllava_load_image(str(item["image"])))

            image_tokens = "<image>\\n" * len(images)
            prompt_parts.append(f"USER: {image_tokens}{''.join(text_parts)} ASSISTANT:")

        inputs: Dict[str, Any] = {"prompt": "".join(prompt_parts)}
        if images:
            inputs["multi_modal_data"] = {
                "image": images[0] if len(images) == 1 else images
            }
        return inputs

    async def _tinyllava_async_chat(
        self,
        messages: List[ChatCompletionMessage],
        generate_config: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        inputs = self._tinyllava_build_inputs(messages)
        logger.info(
            "TinyLLaVA Qwen vLLM adapter fast path, class=%s, prompt=%r, has_image=%s",
            type(self).__name__,
            inputs.get("prompt"),
            "multi_modal_data" in inputs,
        )
        generate_config = self._sanitize_chat_config(generate_config)
        stream = generate_config.get("stream", None)

        if stream:
            agen = await self.async_generate(inputs, generate_config, request_id=request_id)
            assert isinstance(agen, AsyncGenerator)
            return self._async_to_chat_completion_chunks(
                agen, self.reasoning_parser, {}
            )

        c = await self.async_generate(inputs, generate_config, request_id=request_id)
        assert not isinstance(c, AsyncGenerator)
        return self._to_chat_completion(c, self.reasoning_parser)

    async def _gen_tokens_prompt(
        self, tokenizer, prompt: Union[str, dict], config: dict
    ):
        from vllm import TokensPrompt

        if isinstance(prompt, dict):
            prompt_str = prompt["prompt"]
            multi_modal_data = prompt.get("multi_modal_data")
            token_ids = await asyncio.to_thread(
                self._tokenize,
                tokenizer,
                prompt_str,
                config,
            )
            return TokensPrompt(
                prompt_token_ids=token_ids, multi_modal_data=multi_modal_data
            )

        token_ids = await asyncio.to_thread(
            self._tokenize,
            tokenizer,
            prompt,  # type: ignore
            config,
        )
        return TokensPrompt(prompt_token_ids=token_ids)

'''


FAST_PATH = '''        if self.model_family.model_name.startswith(
            "tinyllava-qwen"
        ) or self.model_uid.startswith("tinyllava-qwen"):
            if tools:
                raise ValueError("TinyLLaVA Qwen vLLM adapter does not support tools.")
            return await self._tinyllava_async_chat(
                messages, generate_config, request_id=request_id
            )

'''


MODEL_ACTOR_METHOD = '''    async def tinyllava_chat(self, messages: List[Dict], *args, **kwargs):
        logger.info(
            "ModelActor.tinyllava_chat entered uid=%s model_class=%s",
            self.model_uid(),
            type(self._model).__name__,
        )
        kwargs.pop("raw_params", None)
        if "request_id" not in kwargs:
            kwargs["request_id"] = str(uuid.uuid1())
        else:
            kwargs["request_id"] = str(kwargs["request_id"])
        data = await self._call_wrapper_json(
            self._model.async_chat, messages, *args, **kwargs
        )
        logger.info("ModelActor.tinyllava_chat returned uid=%s", self.model_uid())
        return data

    async def vllm_generate(self, prompt: str, *args, **kwargs):
        logger.info(
            "ModelActor.vllm_generate entered uid=%s model_class=%s",
            self.model_uid(),
            type(self._model).__name__,
        )
        kwargs.pop("raw_params", None)
        if "request_id" not in kwargs:
            kwargs["request_id"] = str(uuid.uuid1())
        else:
            kwargs["request_id"] = str(kwargs["request_id"])
        data = await self._call_wrapper_json(
            self._model.async_generate, prompt, *args, **kwargs
        )
        logger.info("ModelActor.vllm_generate returned uid=%s", self.model_uid())
        return data

    async def vllm_chat(self, messages: List[Dict], *args, **kwargs):
        logger.info(
            "ModelActor.vllm_chat entered uid=%s model_class=%s",
            self.model_uid(),
            type(self._model).__name__,
        )
        kwargs.pop("raw_params", None)
        if "request_id" not in kwargs:
            kwargs["request_id"] = str(uuid.uuid1())
        else:
            kwargs["request_id"] = str(kwargs["request_id"])
        data = await self._call_wrapper_json(
            self._model.async_chat, messages, *args, **kwargs
        )
        logger.info("ModelActor.vllm_chat returned uid=%s", self.model_uid())
        return data

'''


FORCE_TOKENS_PROMPT_BLOCK = '''        prompt_or_token_ids: Union[str, Dict[str, Any], List[int]] = prompt
        if isinstance(prompt, (str, dict)):
            tokenizer = await self._get_tokenizer(lora_request)
            prompt_or_token_ids = await self._gen_tokens_prompt(
                tokenizer,
                prompt,
                sanitized_generate_config,  # type: ignore
            )

        if sampling_params.max_tokens is None:
            if isinstance(prompt_or_token_ids, dict):
                prompt_length = len(prompt_or_token_ids["prompt_token_ids"])
            else:
                prompt_length = len(prompt_or_token_ids)
            sampling_params.max_tokens = max_tokens = self._context_length - prompt_length
            logger.debug("No max_tokens set, setting to: %s", max_tokens)

'''


REST_TINYLLAVA_CALL = '''                if model_uid.startswith("tinyllava-qwen"):
                    data = await model.tinyllava_chat(
                        messages,
                        kwargs,
                        raw_params=raw_kwargs,
                    )
                elif await model.is_vllm_backend():
                    data = await model.vllm_chat(
                        messages,
                        kwargs,
                        raw_params=raw_kwargs,
                    )
                else:
                    data = await model.chat(
                        messages,
                        kwargs,
                        raw_params=raw_kwargs,
                    )
'''


REST_VLLM_GENERATE_CALL = '''                if await model.is_vllm_backend():
                    data = await model.vllm_generate(
                        body.prompt, kwargs, raw_params=raw_kwargs
                    )
                else:
                    data = await model.generate(body.prompt, kwargs, raw_params=raw_kwargs)
'''


def xinference_root() -> Path:
    spec = importlib.util.find_spec("xinference")
    if spec is None or spec.origin is None:
        raise RuntimeError("xinference is not importable; run this inside the uv environment")
    return Path(spec.origin).resolve().parent


def patch_vllm_core(path: Path) -> bool:
    text = path.read_text()
    changed = False

    helper_methods, gen_tokens_tail = TINYLLAVA_BASE_METHODS.split(
        "    async def _gen_tokens_prompt(", 1
    )
    gen_tokens_method = "    async def _gen_tokens_prompt(" + gen_tokens_tail

    if "def _tinyllava_build_inputs" not in text:
        insert_at = text.find("    async def _gen_tokens_prompt(")
        if insert_at == -1:
            raise RuntimeError(f"failed to insert TinyLLaVA base methods into {path}")
        text = text[:insert_at] + helper_methods + text[insert_at:]
        changed = True

    gen_pattern = re.compile(
        r"    async def _gen_tokens_prompt\(\n"
        r".*?"
        r"(?=\n    @vllm_check\n    async def async_generate\()",
        re.S,
    )
    text, count = gen_pattern.subn(gen_tokens_method, text, count=1)
    if count != 1:
        raise RuntimeError(f"failed to patch _gen_tokens_prompt in {path}")
    changed = True

    text, changed_chat = patch_class_fast_path(text, "VLLMChatModel")
    text, changed_multi = patch_class_fast_path(text, "VLLMMultiModel")
    changed = changed or changed_chat or changed_multi

    if "if isinstance(prompt, (str, dict)):" not in text:
        prompt_pattern = re.compile(
            r"        prompt_or_token_ids: Union\[str, Dict\[str, Any\], List\[int\]\] = prompt\n"
            r"        if sampling_params\.max_tokens is None:\n"
            r".*?"
            r"            logger\.debug\(\"No max_tokens set, setting to: %s\", max_tokens\)\n\n",
            re.S,
        )
        text, count = prompt_pattern.subn(FORCE_TOKENS_PROMPT_BLOCK, text, count=1)
        if count != 1:
            raise RuntimeError(f"failed to patch prompt tokenization in {path}")
        changed = True

    if changed:
        path.write_text(text)
    return changed


def patch_class_fast_path(text: str, class_name: str) -> tuple[str, bool]:
    class_start = text.find(f"class {class_name}(")
    if class_start == -1:
        raise RuntimeError(f"{class_name} not found")
    next_class = text.find("\nclass ", class_start + 1)
    class_end = next_class if next_class != -1 else len(text)
    class_text = text[class_start:class_end]
    if "TinyLLaVA Qwen vLLM adapter does not support tools" in class_text:
        return text, False

    needle = (
        "        tools = list(generate_config.pop(\"tools\", [])) if generate_config else None\n"
    )
    idx = class_text.find(needle)
    if idx == -1:
        raise RuntimeError(f"tools extraction line not found in {class_name}")
    insert_at = class_start + idx + len(needle)
    return text[:insert_at] + FAST_PATH + text[insert_at:], True


def patch_model_actor(path: Path) -> bool:
    text = path.read_text()
    if all(
        marker in text
        for marker in [
            "async def tinyllava_chat",
            "async def vllm_generate",
            "async def vllm_chat",
        ]
    ):
        return False

    needle = (
        "    @request_limit\n"
        "    @xo.generator\n"
        "    @log_async(logger=logger)\n"
        "    async def generate(self, prompt: str, *args, **kwargs):\n"
    )
    idx = text.find(needle)
    if idx == -1:
        raise RuntimeError(f"generate method anchor not found in {path}")
    existing_start = text.rfind("    async def tinyllava_chat", 0, idx)
    if existing_start != -1:
        text = text[:existing_start] + text[idx:]
        idx = existing_start
    text = text[:idx] + MODEL_ACTOR_METHOD + text[idx:]
    path.write_text(text)
    return True


def patch_rest_api(path: Path) -> bool:
    text = path.read_text()
    changed = False

    chat_needle = '''                data = await model.chat(
                    messages,
                    kwargs,
                    raw_params=raw_kwargs,
                )
'''
    if "model.vllm_chat" not in text:
        if "model.tinyllava_chat" in text:
            old_tiny_block = '''                if model_uid.startswith("tinyllava-qwen"):
                    data = await model.tinyllava_chat(
                        messages,
                        kwargs,
                        raw_params=raw_kwargs,
                    )
                else:
                    data = await model.chat(
                        messages,
                        kwargs,
                        raw_params=raw_kwargs,
                    )
'''
            if old_tiny_block not in text:
                raise RuntimeError(f"existing TinyLLaVA chat patch not found in {path}")
            text = text.replace(old_tiny_block, REST_TINYLLAVA_CALL, 1)
        else:
            if chat_needle not in text:
                raise RuntimeError(f"chat completion anchor not found in {path}")
            text = text.replace(chat_needle, REST_TINYLLAVA_CALL, 1)
        changed = True

    generate_needle = '''                data = await model.generate(body.prompt, kwargs, raw_params=raw_kwargs)
'''
    if "model.vllm_generate" not in text:
        if generate_needle not in text:
            raise RuntimeError(f"completion anchor not found in {path}")
        text = text.replace(generate_needle, REST_VLLM_GENERATE_CALL, 1)
        changed = True

    if changed:
        path.write_text(text)
    return changed


def main() -> None:
    root = xinference_root()
    changes = {
        "model/llm/vllm/core.py": patch_vllm_core(root / "model/llm/vllm/core.py"),
        "core/model.py": patch_model_actor(root / "core/model.py"),
        "api/restful_api.py": patch_rest_api(root / "api/restful_api.py"),
    }
    for name, changed in changes.items():
        print(f"{name}: {'patched' if changed else 'already patched'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
