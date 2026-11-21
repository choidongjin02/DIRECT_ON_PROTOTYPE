# [가짜 API 묶음] Gemini TTS · OpenAI · TypeSafe(Jev) · Soniox 를 흉내 내는 시험용 서버입니다. 크레딧을 쓰지 않습니다.
# 왜 있나: 서버 코드를 고친 뒤 순서·시간 기록·turn.end 가 맞는지 API 비용 없이 먼저 확인하려고요.
#          지연 시간은 아래 환경변수로 고정하므로, 진짜 API 의 속도를 재는 용도로는 쓰지 않습니다.
"""사용:
  python eval/fake_apis.py                       # HTTP 8290 (Gemini·OpenAI·Jev), WebSocket 8291 (Soniox)
  그다음 상담원 서버를 이 주소로 켭니다 (키는 아무 글자나 넣어도 됨):
    GEMINI_BASE_URL=http://127.0.0.1:8290/ OPENAI_BASE_URL=http://127.0.0.1:8290/v1 \
    TYPESAFE_URL=http://127.0.0.1:8290/v1/systemone SONIOX_URL=ws://127.0.0.1:8291 python agent_server/server.py --port 9300
지연(ms) 설정: FAKE_TTS_FIRST_MS(900) FAKE_TTS_PER_CHAR_MS(8) FAKE_TTS_SPEED(배속 2.5) FAKE_LLM_MS(500) FAKE_JEV_MS(250) FAKE_STT_FINAL_MS(200)
            FAKE_TTS_RPM(0=제한 없음): 분당 요청이 이보다 많으면 429 를 돌려준다
            FAKE_STT_LAG_MS(0): 받아쓰기 중간 결과가 소리보다 이만큼 늦게 따라온다 (진짜 Soniox 처럼 말 끝 직후엔 뒷부분이 빠져 있음)
"""
import argparse, asyncio, base64, hashlib, json, os, re, sys, threading, time, wave
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent_server"))
import case_router                                            # 키워드 분류를 가짜 Jev·LLM 의 '정답'으로 쓴다


def env_ms(name, default):
    return float(os.getenv(name, default)) / 1000


TTS_FIRST, TTS_PER_CHAR, TTS_SPEED = env_ms("FAKE_TTS_FIRST_MS", 900), env_ms("FAKE_TTS_PER_CHAR_MS", 8), float(os.getenv("FAKE_TTS_SPEED", 2.5))
LLM_DELAY, JEV_DELAY, STT_FINAL = env_ms("FAKE_LLM_MS", 500), env_ms("FAKE_JEV_MS", 250), env_ms("FAKE_STT_FINAL_MS", 200)
STT_LAG = env_ms("FAKE_STT_LAG_MS", 0)
TTS_RPM = int(os.getenv("FAKE_TTS_RPM", 0))
TTS_TIMES, LOCK = deque(), threading.Lock()
FAKE_HOLD_ENDS = ("안녕하세요", "여보세요", "저기요", "인데요", "는데요", "은데요", "는데", "인데", "하고", "해서", "아니")
COUNTS = {"tts": 0, "tts_429": 0, "llm": 0, "jev": 0, "stt": 0}


def fake_voice(text):
    """글자 수에 비례하는 길이의 24kHz 음성(사인파). 앞 150ms, 뒤 350ms 는 무음."""
    chars = len(re.sub(r"\s", "", text))
    sec = 0.12 * max(chars, 3)
    t = np.arange(int(24000 * sec)) / 24000
    tone = (6000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    pad = lambda s: np.zeros(int(24000 * s), dtype=np.int16)
    return np.concatenate([pad(0.15), tone, pad(0.35)]).tobytes()


def tts_text(body):
    text = "".join(p.get("text", "") for c in body.get("contents", []) for p in c.get("parts", []))
    return text.split(": ", 1)[1] if ": " in text else text   # "Say: " 같은 지시를 뗀다


def audio_part(pcm):
    return {"candidates": [{"content": {"role": "model", "parts": [{"inlineData": {
        "mimeType": "audio/L16;codec=pcm;rate=24000", "data": base64.b64encode(pcm).decode()}}]}, "finishReason": "STOP"}],
        "modelVersion": "fake-tts"}


def llm_reply(body):
    msgs = body.get("messages", [])
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    last = msgs[-1]
    user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
    if "업무 유형 분류기" in system:                           # case_router 의 LLM 분류
        return {"content": case_router.keyword_case(user.split("[고객]")[-1])}
    if last["role"] == "tool":
        return {"content": "확인해 보니 주문하신 상품은 현재 배송 중입니다. 내일 중으로 도착할 예정입니다."}
    digits = re.findall(r"\d{4,}", user)
    if body.get("tools") and digits and ("주문" in user or "배송" in user):
        return {"tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "get_order_info", "arguments": json.dumps({"order_id": digits[-1]})}}]}
    m = re.search(r"# 2단계 업무별 절차: (.+)", system)
    topic = m.group(1).strip() if m else "문의"
    return {"content": f"{topic} 관련 문의 잘 들었습니다. 주문번호를 알려 주시면 바로 확인해 드리겠습니다."}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"                              # 응답 뒤 연결을 닫는다 (스트리밍 끝을 알리기 쉽다)

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        buf = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(buf)))
        self.end_headers()
        self.wfile.write(buf)

    def do_GET(self):
        if "/models" in self.path:
            return self.send_json({"models": [{"name": "models/gemini-2.5-flash-preview-tts"}]})
        if self.path.startswith("/stats"):
            return self.send_json(COUNTS)
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if "generateContent" in self.path or "streamGenerateContent" in self.path:
            return self.tts(body, stream="streamGenerateContent" in self.path)
        if self.path.endswith("/chat/completions"):
            COUNTS["llm"] += 1
            time.sleep(LLM_DELAY)
            msg = {"role": "assistant", "content": None, **llm_reply(body)}
            return self.send_json({"id": "fake", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
                                   "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
                                   "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
        if self.path.endswith("/systemone"):
            COUNTS["jev"] += 1
            time.sleep(JEV_DELAY)
            if "turn" in body["questions"]:                       # 말 끝 판정: 이어질 말투·인사만 있으면 hold, 아니면 end (무음이 길면 end)
                said = re.sub(r"[\s.,?!…~]+$", "", body["state"]["caller_transcript"])
                hold = said.endswith(FAKE_HOLD_ENDS) and body["state"].get("silence_ms", 0) < 1500
                choice = "hold" if hold else "end"
                return self.send_json({"model": "fake-jev", "answers": {"turn": {"type": "choice", "choice": choice,
                                       "probabilities": {choice: 0.9, "end" if hold else "hold": 0.1}, "confidence": 0.8}}})
            q = body["questions"]["case"]
            choice = case_router.keyword_case(body["state"]["caller_transcript"])
            ids = list(q["criteria"])
            probs = {k: (0.86 if k == choice else 0.14 / (len(ids) - 1)) for k in ids}
            answers = {"case": {"type": "choice", "choice": choice, "probabilities": probs, "confidence": 0.8}}
            if "action" in body["questions"]:
                act = case_router.keyword_action(choice, body["state"]["caller_transcript"], body["state"].get("caller_phone_on_file") == "yes")
                answers["action"] = {"type": "choice", "choice": act, "probabilities": {act: 0.9}, "confidence": 0.8}
            return self.send_json({"model": "fake-jev", "answers": answers, "usage": {"input_tokens": 300, "output_tokens": 1}})
        self.send_json({"error": "not found"}, 404)

    def tts(self, body, stream):
        with LOCK:
            now = time.time()
            while TTS_TIMES and now - TTS_TIMES[0] > 60:
                TTS_TIMES.popleft()
            if TTS_RPM and len(TTS_TIMES) >= TTS_RPM:
                COUNTS["tts_429"] += 1
                return self.send_json({"error": {"code": 429, "message": "Resource has been exhausted (fake).", "status": "RESOURCE_EXHAUSTED"}}, 429)
            TTS_TIMES.append(now)
            COUNTS["tts"] += 1
        text = tts_text(body)
        pcm = fake_voice(text)
        first = TTS_FIRST + TTS_PER_CHAR * len(text)
        audio_sec = len(pcm) / 48000
        if not stream:
            time.sleep(first + audio_sec / TTS_SPEED)
            return self.send_json(audio_part(pcm))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(first)
        step = 24000 * 2 // 2                                  # 0.5초 분량씩
        for i in range(0, len(pcm), step):
            if i:
                time.sleep(0.5 / TTS_SPEED)
            self.wfile.write(f"data: {json.dumps(audio_part(pcm[i:i + step]))}\r\n\r\n".encode())
            self.wfile.flush()


# ── 가짜 Soniox: 녹음 파일 앞 1초로 어떤 문항인지 알아내 정해진 글을 돌려준다 ──
def transcript_table():
    """녹음 앞 1초의 해시 → (대본, 말이 끝나는 시각 초)."""
    table = {}
    for cases, audio in (("data/ko_cases.jsonl", "data/audio/ko"), ("data/demo_cases.jsonl", "data/audio/demo")):
        path = ROOT / cases
        if not path.exists():
            continue
        meta = json.loads((ROOT / audio / "meta.json").read_text(encoding="utf-8")) if (ROOT / audio / "meta.json").exists() else {}
        for line in path.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            wav = ROOT / audio / f"{c['case_id']}.wav"
            if wav.exists():
                with wave.open(str(wav)) as w:
                    table[hashlib.sha1(w.readframes(16000)).hexdigest()] = (re.sub(r"\s+", " ", c["customer_utterance"]),
                                                                            meta.get(c["case_id"], {}).get("speech_end", 0))
    return table


TABLE = transcript_table()


async def soniox(ws):
    COUNTS["stt"] += 1
    await ws.recv()                                            # 설정 JSON
    audio, text, end, n = b"", None, 0, 0
    async for m in ws:
        if isinstance(m, bytes):
            audio += m
            n += 1
            if text is None and len(audio) >= 32000:
                text, end = TABLE.get(hashlib.sha1(audio[:32000]).hexdigest(), ("알 수 없는 음성입니다", 0))
            if text and n % 5 == 0:                            # 100ms 마다 아직 확정 전인 글을 보낸다
                heard = len(audio) / 32000 - STT_LAG           # 받아쓰기가 따라온 소리 위치(초)
                shown = text if not end else text[: max(0, min(len(text), round(len(text) * heard / end)))]
                await ws.send(json.dumps({"tokens": [{"text": shown, "is_final": False}]}))
        elif json.loads(m).get("type") == "finalize":
            await asyncio.sleep(STT_FINAL)
            await ws.send(json.dumps({"tokens": [{"text": text or "", "is_final": True}, {"text": "<fin>", "is_final": True}]}))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http-port", type=int, default=8290)
    ap.add_argument("--ws-port", type=int, default=8291)
    a = ap.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", a.http_port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    async with websockets.serve(soniox, "127.0.0.1", a.ws_port, max_size=None):
        print(f"가짜 API: HTTP {a.http_port} (Gemini·OpenAI·Jev), Soniox ws {a.ws_port}, 음성 대본 {len(TABLE)}개", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
