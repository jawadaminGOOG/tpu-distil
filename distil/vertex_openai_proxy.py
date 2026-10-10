#!/usr/bin/env python3
"""OpenAI-compatible local proxy in front of a Vertex AI dedicated endpoint.

Why: OpenAI-compatible clients need a static base URL + key; Vertex needs a
short-lived OAuth bearer token. This proxy injects a fresh token (from the GCE
metadata server) and forwards request bodies byte-for-byte. It adds nothing to
prompts or responses.

/v1/models is answered locally with the served model name and the
max_model_len the container was launched with (passed via --max-model-len,
must equal the container's --max-model-len flag).

Usage: vertex_openai_proxy.py --target-base <https://.../endpoints/ID> \
         --model deepseek-ai/DeepSeek-V4.1-Flash --max-model-len 65536 --port 18100
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_TOKEN = {"value": None, "exp": 0.0}
_LOCK = threading.Lock()
_MD = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"


def token() -> str:
    with _LOCK:
        if time.time() > _TOKEN["exp"] - 300:
            req = urllib.request.Request(_MD, headers={"Metadata-Flavor": "Google"})
            d = json.load(urllib.request.urlopen(req, timeout=10))
            _TOKEN["value"] = d["access_token"]
            _TOKEN["exp"] = time.time() + int(d["expires_in"])
        return _TOKEN["value"]


def make_handler(args):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                body = {"object": "list", "data": [{
                    "id": args.model, "object": "model", "owned_by": "vertex",
                    "max_model_len": args.max_model_len}]}
                return self._send(200, json.dumps(body).encode())
            if self.path.rstrip("/") in ("/health", "/ping"):
                return self._send(200, b"{}")
            return self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(n)
            sub = self.path[len("/v1"):] if self.path.startswith("/v1") else self.path
            if args.predict_mode:
                if sub.rstrip("/") != "/chat/completions":
                    return self._send(404, b'{"error":"only /v1/chat/completions is supported"}')
                body = json.loads(data)
                if body.get("stream"):
                    return self._send(400, b'{"error":"stream not supported by proxy"}')
                body["@requestFormat"] = "chatCompletions"
                url = args.target_base.rstrip("/") + ":predict"
                data = json.dumps({"instances": [body]}).encode()
            else:
                url = args.target_base.rstrip("/") + sub
            req = urllib.request.Request(url, data=data, method="POST", headers={
                "Authorization": f"Bearer {token()}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=args.timeout) as r:
                    raw = r.read()
                    if not args.predict_mode:
                        return self._send(r.status, raw, r.headers.get("Content-Type", "application/json"))
                    out = json.loads(raw)["predictions"]
                    if args.merge_reasoning:
                        # Inverse of the server's reasoning parser: restore the raw
                        # completion text, which is what the parser-free TPU servers
                        # return for every other arm. The chat template ends the prompt
                        # with <think>, so a completed output always contains </think>
                        # (as its first token when the model skips thinking; the parser
                        # then reports reasoning=None). A length-truncated output with
                        # no </think> is returned by the parser as reasoning only.
                        for ch in out.get("choices", []):
                            m = ch.get("message") or {}
                            rs = m.get("reasoning")
                            if rs is None:
                                rs = m.get("reasoning_content")
                            ct = m.get("content")
                            if ch.get("finish_reason") == "length" and not ct:
                                m["content"] = rs or ""
                            else:
                                m["content"] = (rs or "") + "</think>" + (ct or "")
                            m["reasoning"] = None
                            m.pop("reasoning_content", None)
                    for k in ("prompt_token_ids", "prompt_text", "prompt_logprobs",
                              "kv_transfer_params", "ec_transfer_params"):
                        out.pop(k, None)
                    return self._send(200, json.dumps(out).encode())
            except urllib.error.HTTPError as e:
                eb = e.read()
                if args.verbose:
                    print(f"upstream HTTP {e.code}: {eb[:500]!r}", flush=True)
                if args.predict_mode:
                    try:
                        err = json.loads(eb).get("error") or {}
                    except ValueError:
                        err = {}
                    msg = err.get("message", "") if isinstance(err, dict) else ""
                    if "maximum context length" in msg:
                        # Vertex wraps vLLM's 400 context-overflow as 500. Restore
                        # vLLM's native status/shape so OpenAI-compatible clients see
                        # the same context-length error as from a plain vLLM server.
                        body = {"error": {"message": msg, "type": "BadRequestError",
                                          "param": err.get("param") or "input_tokens", "code": 400}}
                        return self._send(400, json.dumps(body).encode())
                return self._send(e.code, eb)
            except Exception as e:  # network error: surface as 502, never fabricate a completion
                return self._send(502, json.dumps({"error": repr(e)}).encode())

        def log_message(self, fmt, *a):
            if args.verbose:
                super().log_message(fmt, *a)

    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--port", type=int, default=18100)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--predict-mode", action="store_true",
                   help="wrap OpenAI chat bodies into Vertex :predict instances (Model Garden api_server)")
    p.add_argument("--merge-reasoning", action="store_true",
                   help="re-join reasoning + '</think>' + content (server runs a reasoning parser)")
    args = p.parse_args()
    ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(args)).serve_forever()


if __name__ == "__main__":
    main()
