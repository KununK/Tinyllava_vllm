#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "run"
LOG_DIR = RUN_DIR / "logs"
PID_FILE = RUN_DIR / "xinference.pid"


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def is_local_model_path(value: str) -> bool:
    return value.startswith(("/", "./", "../")) or Path(value).exists()


def model_kind(model: dict[str, Any]) -> str:
    return str(model.get("kind", "tinyllava"))


def model_path_arg(model: dict[str, Any]) -> str | None:
    path = str(model.get("model_path", ""))
    if not path:
        return None
    if model_kind(model) == "text" and not is_local_model_path(path):
        return None
    return str(repo_path(path))


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        config = json.load(f)
    config.setdefault("server", {})
    config.setdefault("models", [])
    return config


def enabled_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [model for model in config["models"] if model.get("enabled", True)]


def local_env(config: dict[str, Any]) -> dict[str, str]:
    server = config["server"]
    env = os.environ.copy()
    env["PATH"] = f"{ROOT / '.venv/bin'}:{env.get('PATH', '')}"
    env["NO_PROXY"] = ",".join(
        part
        for part in [env.get("NO_PROXY"), "127.0.0.1", "localhost"]
        if part
    )
    env["XINFERENCE_HOST"] = str(server.get("host", "0.0.0.0"))
    env["XINFERENCE_PORT"] = str(server.get("port", 10003))
    env["XINFERENCE_HOME"] = str(repo_path(server.get("xinference_home", ".xinference_multi")))
    env.setdefault("XINFERENCE_MODEL_SRC", "huggingface")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    return env


def endpoint(config: dict[str, Any]) -> str:
    server = config["server"]
    return str(server.get("endpoint") or f"http://127.0.0.1:{server.get('port', 10003)}").rstrip("/")


def run_cmd(cmd: list[str], env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env, text=True, check=check)


def is_server_ready(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=3)
        return response.status_code < 500
    except requests.RequestException:
        return False


def wait_server(base_url: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_server_ready(base_url):
            return
        time.sleep(2)
    raise TimeoutError(f"Xinference did not become ready at {base_url} in {timeout_seconds}s")


def start_server(config: dict[str, Any], env: dict[str, str]) -> None:
    base_url = endpoint(config)
    if is_server_ready(base_url):
        print(f"Xinference is already available: {base_url}")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "xinference.log"
    log_file = log_path.open("ab")
    proc = subprocess.Popen(
        [str(ROOT / "scripts/start_xinference.sh")],
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(f"{proc.pid}\n")
    print(f"Started Xinference pid={proc.pid}, log={log_path}")
    wait_server(base_url, int(config["server"].get("startup_timeout_seconds", 90)))


def stop_server(config: dict[str, Any], env: dict[str, str]) -> None:
    if is_server_ready(endpoint(config)):
        terminate_models(config, env)
    if not PID_FILE.exists():
        print(f"No pid file: {PID_FILE}")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    PID_FILE.unlink(missing_ok=True)
    print(f"Stopped Xinference pid={pid}")


def register_models(config: dict[str, Any], env: dict[str, str]) -> None:
    base_url = endpoint(config)
    xinference = str(ROOT / ".venv/bin/xinference")
    for model in enabled_models(config):
        if not model.get("register_file"):
            print(f"Skip custom register for builtin model: {model['model_name']}")
            continue
        model_name = model["model_name"]
        register_file = repo_path(model["register_file"])
        run_cmd(
            [
                xinference,
                "unregister",
                "--endpoint",
                base_url,
                "--model-type",
                "LLM",
                "--model-name",
                model_name,
            ],
            env=env,
            check=False,
        )
        run_cmd(
            [
                xinference,
                "register",
                "--endpoint",
                base_url,
                "--model-type",
                "LLM",
                "--file",
                str(register_file),
                "--persist",
            ],
            env=env,
        )


def loaded_model_uids(base_url: str) -> set[str]:
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return set()
    return {item.get("id") for item in data.get("data", []) if item.get("id")}


def wait_model_loaded(base_url: str, model_uid: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if model_uid in loaded_model_uids(base_url):
            return
        time.sleep(2)
    raise TimeoutError(
        f"Model {model_uid} was not loaded at {base_url} in {timeout_seconds}s. "
        f"Check {LOG_DIR / 'xinference.log'} for the backend error."
    )


def terminate_models(config: dict[str, Any], env: dict[str, str]) -> None:
    base_url = endpoint(config)
    xinference = str(ROOT / ".venv/bin/xinference")
    loaded = loaded_model_uids(base_url)
    for model in enabled_models(config):
        if model["model_uid"] not in loaded:
            print(f"Model not loaded, skip terminate: {model['model_uid']}")
            continue
        run_cmd(
            [
                xinference,
                "terminate",
                "--endpoint",
                base_url,
                "--model-uid",
                model["model_uid"],
            ],
            env=env,
            check=False,
        )


def launch_models(config: dict[str, Any], env: dict[str, str], restart_models: bool) -> None:
    base_url = endpoint(config)
    if restart_models:
        terminate_models(config, env)
        time.sleep(2)

    loaded = loaded_model_uids(base_url)
    xinference = str(ROOT / ".venv/bin/xinference")
    for model in enabled_models(config):
        if model["model_uid"] in loaded:
            print(f"Model already loaded: {model['model_uid']}")
            continue

        path_arg = model_path_arg(model)
        cmd = [
            xinference,
            "launch",
            "--endpoint",
            base_url,
            "--model-name",
            model["model_name"],
            "--model-uid",
            model["model_uid"],
            "--model-type",
            "LLM",
            "--model-engine",
            "vLLM",
            "--model-format",
            str(model.get("model_format", "pytorch")),
            "--size-in-billions",
            str(model["size_in_billions"]),
            "--quantization",
            str(model.get("quantization", "none")),
            "--n-gpu",
            "1",
            "--gpu-idx",
            str(model["gpu_idx"]),
            "--trust-remote-code",
            str(model.get("trust_remote_code", False)).lower(),
            "--disable-virtual-env",
            "--max_model_len",
            str(model.get("context_length", 1024)),
            "--gpu_memory_utilization",
            str(model.get("gpu_memory_utilization", 0.3)),
        ]
        if path_arg:
            cmd.extend(["--model-path", path_arg])
        elif model.get("model_path"):
            print(f"Use Xinference builtin source for {model['alias']}: {model['model_path']}")
        if "limit_mm_per_prompt" in model:
            cmd.extend(
                [
                    "--limit_mm_per_prompt",
                    json.dumps(model["limit_mm_per_prompt"]),
                ]
            )
        if "enforce_eager" in model:
            cmd.extend(
                [
                    "--enforce_eager",
                    str(model["enforce_eager"]).lower(),
                ]
            )
        for key, value in model.get("vllm_args", {}).items():
            cmd.extend([f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)])
        run_cmd(cmd, env=env)
        wait_model_loaded(
            base_url,
            model["model_uid"],
            int(model.get("load_timeout_seconds", config["server"].get("model_load_timeout_seconds", 300))),
        )
        loaded.add(model["model_uid"])


def print_api_summary(config: dict[str, Any]) -> None:
    base_url = endpoint(config)
    print("\nAPIs:")
    for model in enabled_models(config):
        print(f"  {model['alias']}:")
        print(f"    POST {base_url}/v1/chat/completions")
        print(f"    model: {model['model_uid']}")
        print(f"    gpu: {model['gpu_idx']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=RUN_DIR / "models.json")
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--skip-register", action="store_true")
    parser.add_argument("--skip-launch", action="store_true")
    parser.add_argument("--terminate-models", action="store_true")
    parser.add_argument("--restart-models", action="store_true")
    parser.add_argument("--stop-server", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    env = local_env(config)

    if args.stop_server:
        stop_server(config, env)
        return

    if args.terminate_models:
        wait_server(endpoint(config), int(config["server"].get("startup_timeout_seconds", 90)))
        terminate_models(config, env)
        return

    run_cmd([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/patch_xinference_tinyllava_qwen.py")], env=env)

    if not args.no_start_server:
        start_server(config, env)
    else:
        wait_server(endpoint(config), int(config["server"].get("startup_timeout_seconds", 90)))

    if not args.skip_register:
        register_models(config, env)
    if not args.skip_launch:
        launch_models(config, env, restart_models=args.restart_models)
    print_api_summary(config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
