"""llama-server process management + streaming OpenAI-style chat client with tool calls."""

import hashlib
import json
import os
import subprocess
import time

import requests

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Warmed-up prompt caches are saved here, so a restart restores them in seconds instead of
# re-processing every system prompt (which took most of the 1-2 min startup).
KV_CACHE_DIR = os.path.join(ROOT, "logs", "kv-cache")


def server_cmd(cfg):
    os.makedirs(KV_CACHE_DIR, exist_ok=True)   # llama-server refuses a --slot-save-path that doesn't exist
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
        "--slot-save-path", KV_CACHE_DIR,  # warm-up caches saved to / restored from disk
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


CHOICE_FILE = os.path.join(ROOT, "logs", "brain.txt")


class Brain:
    """Which model is running. Only one llama-server at a time (two would fight over CPU and RAM);
    switching stops the current one and starts the other on the same port."""

    def __init__(self, log_path, log=print):
        self.log_path = log_path
        self.log = log
        self.proc = None
        self.name = self._saved()

    @staticmethod
    def _saved():
        try:
            with open(CHOICE_FILE) as f:
                name = f.read().strip()
        except OSError:
            name = ""
        return name if name in config.BRAINS and _installed(name) else config.BRAIN

    @property
    def cfg(self):
        return config.BRAINS[self.name]

    @property
    def label(self):
        return self.cfg.get("label", self.name)

    def options(self):
        """Installed brains, in config order."""
        return [n for n in config.BRAINS if _installed(n)]

    def next_name(self):
        opts = self.options()
        return opts[(opts.index(self.name) + 1) % len(opts)] if self.name in opts and len(opts) > 1 else None

    def start(self):
        if is_ready(self.cfg["port"]):          # left over from a crashed run: we can't tell which model
            self._kill_port(self.cfg["port"])
        log = open(self.log_path, "ab")
        log.write(f"\n===== starting brain: {self.label} =====\n".encode())
        log.flush()
        self.proc = subprocess.Popen(server_cmd(self.cfg), stdout=log, stderr=subprocess.STDOUT)
        wait_ready(self.cfg["port"], timeout=300, proc=self.proc)

    def stop(self):
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(5)

    def switch(self, name):
        if name not in config.BRAINS or not _installed(name):
            raise ValueError(f"brain '{name}' isn't installed")
        self.stop()
        self.name = name
        try:
            os.makedirs(os.path.dirname(CHOICE_FILE), exist_ok=True)
            with open(CHOICE_FILE, "w") as f:
                f.write(name + "\n")
        except OSError:
            pass
        self.start()

    @staticmethod
    def _kill_port(port):
        subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end = time.monotonic() + 10
        while is_ready(port) and time.monotonic() < end:
            time.sleep(0.3)


def _installed(name):
    return os.path.exists(os.path.join(ROOT, config.BRAINS[name]["model"]))


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

    def _slot_url(self, slot, action):
        base = self.url.rsplit("/v1/", 1)[0]
        return f"{base}/slots/{slot}?action={action}"

    def warm_up_cached(self, system_prompt, tools, slot, model, log=print):
        """Like warm_up, but restore the slot's cache from disk when the exact same model, prompt
        and tools were warmed up before; otherwise warm up and save it for next time.
        Returns "restored" or "warmed"."""
        key = hashlib.sha1(json.dumps([os.path.basename(model), system_prompt, tools],
                                      sort_keys=True).encode()).hexdigest()[:16]
        name = f"slot{slot}-{key}.bin"
        path = os.path.join(KV_CACHE_DIR, name)
        if os.path.exists(path):
            try:
                r = self.session.post(self._slot_url(slot, "restore"), json={"filename": name}, timeout=60)
                if r.status_code == 200:
                    return "restored"
                log(f"[llm] cache restore for slot {slot} failed ({r.status_code}), warming up instead")
            except requests.RequestException as e:
                log(f"[llm] cache restore for slot {slot} failed ({e}), warming up instead")
        self.warm_up(system_prompt, tools, slot)
        try:
            os.makedirs(KV_CACHE_DIR, exist_ok=True)
            r = self.session.post(self._slot_url(slot, "save"), json={"filename": name}, timeout=60)
            if r.status_code == 200:
                for old in os.listdir(KV_CACHE_DIR):      # drop this slot's outdated caches
                    if old.startswith(f"slot{slot}-") and old != name:
                        os.remove(os.path.join(KV_CACHE_DIR, old))
            else:
                log(f"[llm] cache save for slot {slot} failed ({r.status_code})")
        except (requests.RequestException, OSError) as e:
            log(f"[llm] cache save for slot {slot} failed ({e})")
        return "warmed"

    def warm_up(self, system_prompt, tools, slot=None):
        """Process the (static) system prompt + tool list once so it sits in the KV cache.
        Every later request only has to process the new conversation tokens."""
        for _ in self.stream([{"role": "system", "content": system_prompt},
                              {"role": "user", "content": "hi"}], tools=tools, max_tokens=1, slot=slot):
            pass
