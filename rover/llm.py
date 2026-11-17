"""llama-server process management + streaming OpenAI-style chat client with tool calls."""

import json
import os
import subprocess
import time

import requests

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def server_cmd(cfg):
    return [
        config.LLAMA_SERVER,
        "-m", os.path.join(ROOT, cfg["model"]),
        "--port", str(cfg["port"]), "--host", "127.0.0.1",
        "-c", str(cfg["ctx"]),
        "-t", str(config.LLM_THREADS_GEN), "-tb", str(config.LLM_THREADS_BATCH),
        "--jinja",                 # needed for native tool calling
        "--load-mode", "none",     # read weights into RAM (mmap pages got evicted and re-read from SD)
        "-np", str(cfg.get("slots", 1)),  # slots each keep their own cached prompt prefix
        "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",   # 8-bit KV cache: half the RAM
        "--reasoning-budget", "0",  # no hidden "thinking" tokens: answers start right away
    ]


def start_server(cfg, log_path):
    """Start llama-server unless one is already answering on that port."""
    if is_ready(cfg["port"]):
        return None
    log = open(log_path, "ab")
    return subprocess.Popen(server_cmd(cfg), stdout=log, stderr=subprocess.STDOUT)


def is_ready(port):
    try:
        return requests.get(f"http://127.0.0.1:{port}/health", timeout=0.5).status_code == 200
    except requests.RequestException:
        return False


def wait_ready(port, timeout=180, proc=None):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if is_ready(port):
            return True
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"llama-server on port {port} exited with code {proc.returncode}")
        time.sleep(0.5)
    raise TimeoutError(f"llama-server on port {port} not ready after {timeout}s")


class ChatClient:
    def __init__(self, cfg):
        self.url = f"http://127.0.0.1:{cfg['port']}/v1/chat/completions"
        self.session = requests.Session()
        self._resp = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        r = self._resp
        if r is not None:
            try:
                r.close()
            except Exception:
                pass

    def stream(self, messages, tools=None, temperature=0.7, max_tokens=400, slot=None, tool_choice=None):
        """Yield ("text", delta) as tokens arrive, then ("tool_calls", [...]) if any,
        then ("done", {"text": full_text, "timings": {...}})."""
        body = {
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
        if tools and tool_choice:
            body["tool_choice"] = tool_choice   # "required" = must call a tool
        if slot is not None:
            body["id_slot"] = slot  # pin to a slot so its cached prompt prefix gets reused
        text, calls, timings = [], {}, {"finish_reason": None}
        self._cancelled = False
        try:
            yield from self._read(body, text, calls, timings)
        except Exception:
            if not self._cancelled:   # closing the stream from cancel() makes requests throw
                raise
            return
        finally:
            self._resp = None
        if self._cancelled:
            return
        if calls:
            out = []
            for i in sorted(calls):
                c = calls[i]
                try:
                    args = json.loads(c["args"]) if c["args"].strip() else {}
                except json.JSONDecodeError:
                    args = {}
                out.append({"id": c["id"] or f"call_{i}", "name": c["name"], "args": args, "raw": c["args"]})
            yield "tool_calls", out
        yield "done", {"text": "".join(text), "timings": timings}

    def _read(self, body, text, calls, timings):
        with self.session.post(self.url, json=body, stream=True, timeout=(5, 120)) as r:
            self._resp = r
            r.raise_for_status()
            r.encoding = "utf-8"  # SSE has no charset header; requests would guess latin-1
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                data = raw[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                timings.update(chunk.get("timings") or {})
                for choice in chunk.get("choices", []):
                    if choice.get("finish_reason"):
                        timings["finish_reason"] = choice["finish_reason"]   # "length" = cut off
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        text.append(delta["content"])
                        yield "text", delta["content"]
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function", {})
                        slot["name"] += fn.get("name") or ""
                        slot["args"] += fn.get("arguments") or ""

    def warm_up(self, system_prompt, tools, slot=None):
        """Process the (static) system prompt + tool list once so it sits in the KV cache.
        Every later request only has to process the new conversation tokens."""
        for _ in self.stream([{"role": "system", "content": system_prompt},
                              {"role": "user", "content": "hi"}], tools=tools, max_tokens=1, slot=slot):
            pass
