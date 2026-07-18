"""CPU embedding server: Qwen3-Embedding-0.6B via transformers, OpenAI-compatible.

왜: 이 vLLM 버전은 (a) 부팅 사전검사 free>=util*total, (b) KV 계산이 타 프로세스
포함이라 24GB에서 9B+임베더+리랭커 3개 동시 상주가 불가능하다. 쿼리 임베딩은
요청당 1건이라 CPU로 충분 -> 임베더만 CPU로 내려 GPU를 9B+리랭커에 양보한다.

vLLM /v1/embeddings와 같은 계약({model, input[]} -> {data[{index, embedding}]}).
Last-token pooling + L2 normalize (Qwen3-Embedding 공식 방식). GPU 임베더와의
벡터 일치는 scripts로 검증(코사인 > 0.999 확인 후 사용).

    ~/miniconda3/envs/vllm/bin/python scripts/cpu_embed_server.py --port 8011
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


class Embedder:
    def __init__(self) -> None:
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        self.model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> list[list[float]]:
        batch = self.tok(texts, padding=True, truncation=True, max_length=8192,
                         return_tensors="pt")
        out = self.model(**batch)
        # left padding -> last position is the last real token
        emb = out.last_hidden_state[:, -1]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()
    emb = Embedder()
    print("model loaded (cpu)", flush=True)

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            texts = body["input"]
            if isinstance(texts, str):
                texts = [texts]
            vecs = emb.encode(texts)
            resp = {"object": "list",
                    "data": [{"object": "embedding", "index": i, "embedding": v}
                             for i, v in enumerate(vecs)],
                    "model": body.get("model", MODEL_ID)}
            payload = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802 -- health check
            payload = json.dumps({"data": [{"id": "qwen3-emb-cpu"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):  # quiet
            pass

    print(f"serving on 0.0.0.0:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
