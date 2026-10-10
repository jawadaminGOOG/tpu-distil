"""bench_serving.run_level against an in-process fake OpenAI server."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DISTIL = Path(__file__).resolve().parents[1] / "distil"
sys.path.insert(0, str(DISTIL))
import bench_serving  # noqa: E402


class _H(BaseHTTPRequestHandler):
  def do_POST(self):  # noqa: N802
    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
    if body["messages"][-1]["content"] == "fail":
      self.send_response(400)
      self.end_headers()
      self.wfile.write(b"bad")
      return
    out = json.dumps({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
    self.send_response(200)
    self.send_header("Content-Length", str(len(out)))
    self.end_headers()
    self.wfile.write(out)

  def log_message(self, *a):
    pass


def test_run_level_counts_tokens_and_errors():
  srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  try:
    rows = [{"id": str(i), "messages": [{"role": "user", "content": "fail" if i == 3 else "q"}]} for i in range(8)]
    lv = bench_serving.run_level(rows, {"model": "m", "max_tokens": 5}, f"http://127.0.0.1:{srv.server_port}/v1",
                                 4, timeout=10)
  finally:
    srv.shutdown()
  assert lv["requests"] == 8 and lv["ok"] == 7 and lv["errors"] == 1
  assert lv["mean_completion_tokens"] == 5.0
  assert abs(lv["output_tok_per_s"] * lv["wall_s"] - 35) / 35 < 0.1  # wall_s is rounded to ms
  assert lv["latency_p50_s"] <= lv["latency_p90_s"] <= lv["latency_p99_s"]
