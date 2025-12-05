import argparse
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

from whisper import Whisper
import cgi


# 模型缓存：避免每次请求都重新加载
_model_cache = {}

def get_model(model_type, model_path, language, task):
    key = (model_type, model_path, language, task)
    if key not in _model_cache:
        print(f"Loading model: type={model_type}, path={model_path}, lang={language}, task={task}")
        _model_cache[key] = Whisper(model_type, model_path, language, task)
    return _model_cache[key]


class WhisperHandler(BaseHTTPRequestHandler):

    def _send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/asr":
            self._send_json({"error": "not found"}, 404)
            return

        # 解析 multipart/form-data
        content_type = self.headers.get('Content-Type')
        if not content_type:
            self._send_json({"error": "Missing Content-Type"}, 400)
            return

        ctype, pdict = cgi.parse_header(content_type)

        if ctype != 'multipart/form-data':
            self._send_json({"error": "Only multipart/form-data is supported"}, 400)
            return

        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
        pdict['CONTENT-LENGTH'] = int(self.headers['Content-Length'])

        form = cgi.parse_multipart(self.rfile, pdict)

        # 必须包含 wav 文件
        if "wav" not in form:
            self._send_json({"error": "Field 'wav' is required"}, 400)
            return

        # 获取参数（如果缺省则使用默认值）
        model_type = form.get("model_type", ["tiny"])[0]
        model_path = form.get("model_path", ["../models/models-ax650"])[0]
        language = form.get("language", ["zh"])[0]
        task = form.get("task", ["transcribe"])[0]

        if task not in ("transcribe", "translate"):
            self._send_json({"error": "task must be 'transcribe' or 'translate'"}, 400)
            return

        wav_bytes = form["wav"][0]

        # 写入临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            wav_path = tmp.name

        # 加载模型并运行
        try:
            model = get_model(model_type, model_path, language, task)
            result_text = model.run(wav_path)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)
            return
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

        self._send_json({"text": result_text})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    port = args.port
    server = HTTPServer(("0.0.0.0", port), WhisperHandler)
    print(f"Server started at http://0.0.0.0:{port}")
    server.serve_forever()