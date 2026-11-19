# [AI 상담원 서버] 고객 목소리를 듣고(STT) → 생각하고(LLM, 필요하면 전산실 조회) → 목소리로 답하는(TTS) 본체입니다 (9000번).
# 왜 있나: 대회에서 우리가 만들어야 하는 핵심이에요. 말 끝 판단, 문지기, 끝 신호(turn.end) 안전망까지 맡아요.
# 매뉴얼(말투·규칙)은 같은 폴더의 prompt.md 에, API 키는 같은 폴더의 .env 에 적습니다.
"""DIRECT:ON 2026 AI 상담원 서버 (WebSocket, 기본 9000번 포트).

전화 한 통 = 연결 한 번:
  듣기(말 끝 판단, 목소리는 Soniox 로 실시간 전달) → 받아쓰기 확정 → 생각(LLM + 모의 함수 서버) → 말하기(TTS) → turn.end

대회 지정 모델:  STT = Soniox stt-rt-v6 / LLM = gpt-5.4-mini / TTS = Gemini 2.5 Flash TTS
실행:  python agent_server/server.py --port 9000
준비:  pip install -r agent_server/requirements.txt
       agent_server/.env 에 OPENAI_API_KEY, SONIOX_API_KEY, GEMINI_API_KEY 를 적는다 (.env.example 을 복사해서 사용)
"""
import argparse, asyncio, json, math, os, re, time
from contextlib import suppress
from pathlib import Path

import httpx, numpy as np, websockets
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from openai import AsyncOpenAI
from scipy.signal import resample_poly

load_dotenv(Path(__file__).with_name(".env"), encoding="utf-8-sig")   # agent_server/.env 의 키를 읽는다 (메모장 BOM 도 OK, 이미 설정된 환경변수가 우선)
missing = [k for k in ("OPENAI_API_KEY", "SONIOX_API_KEY", "GEMINI_API_KEY") if not os.getenv(k)]
if missing:
    raise SystemExit(f"API 키가 없습니다: {', '.join(missing)}\n→ agent_server/.env 파일에 한 줄씩 적어 주세요. 예) OPENAI_API_KEY=키값")

# ── 설정 ────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
LLM_EFFORT = os.getenv("LLM_EFFORT")                      # 생각 강도 none / low / medium ... (비우면 모델 기본값=none)
SONIOX_KEY = os.environ["SONIOX_API_KEY"]
SONIOX_URL = os.getenv("SONIOX_URL", "wss://stt-rt.soniox.com/transcribe-websocket")
SONIOX_CONFIG = {"model": "stt-rt-v6", "audio_format": "pcm_s16le", "sample_rate": 16000, "num_channels": 1,
                 "language_hints": ["ko"]}               # 말 끝은 우리 눈치가 정하고 finalize 를 보낸다 (Soniox 자체 endpoint 감지는 끔)
STT_TIMEOUT_S = 5                                         # finalize 뒤 확정 글자를 기다리는 최대 시간
TTS_MODEL = "gemini-2.5-flash-preview-tts"
TTS_VOICE = os.getenv("TTS_VOICE", "Kore")
TTS_STYLE = os.getenv("TTS_STYLE", "Say: ")               # 읽을 글 앞에 붙이는 지시. 없으면 Gemini 가 글을 '질문'으로 착각해 실패한다 (예: "차분하게 말해 주세요: ")
TTS_KO_STYLE = "다음 문장을 그대로 읽어 주세요: "                # 위 지시가 실패했을 때 쓰는 한국어 지시 (직접 시험했을 때 성공한 문구)
TTS_RETRY_STYLE = "Read the following text aloud exactly as written: "   # 첫 시도가 실패했을 때 쓰는 더 분명한 지시
TTS_STREAM = os.getenv("TTS_STREAM", "1") == "1"            # 1 이면 음성을 조각으로 받는 대로 바로 내보낸다 (첫 소리가 빨라짐). 안 되면 자동으로 통째 방식으로 돌아간다
TTS_PER_SENTENCE = os.getenv("TTS_PER_SENTENCE", "0") == "1"   # 0 이면 답 전체를 한 번에 읽힌다 (요청 수가 줄어 무료 등급 한도에 유리)
TTS_RATE = 24000                                          # Gemini TTS 출력은 24kHz 16-bit mono. 목소리가 이상하면 여기부터 확인

RATE, FRAME = 16000, 640            # 대회 규격: 16kHz 16-bit mono, 20ms = 640바이트
SPEECH_RMS = int(os.getenv("SPEECH_RMS", 500))            # 볼륨이 이보다 크면 말소리
END_SILENCE_MS = int(os.getenv("END_SILENCE_MS", 800))    # 말소리 뒤 이만큼 조용하면 말이 끝난 것
MIN_SPEECH_MS = 200                 # 이보다 짧은 소리(잡음)는 말로 치지 않는다
LONG_SILENCE_MS = int(os.getenv("LONG_SILENCE_MS", 1800))  # 말이 '~는데', '~하고' 처럼 이어질 모양으로 멈췄을 땐 이만큼 더 기다린다 (END_SILENCE_MS 와 같게 두면 끔)
CONTINUING = ("는데", "은데", "인데", "하고", "했고", "였고", "이고", "있고", "없고", "싶고", "해서", "어서", "아서", "라서",
              "하면", "니까", "지만", "으며", "하며", "그리고", "그런데", "근데", "그래서")
MAX_LISTEN_MS = 30_000              # 아무리 길어도 여기서 듣기를 멈춘다
LEAD_RMS, LEAD_KEEP_MS = 300, 40    # 말하기 음성 앞 무음을 자를 때: 이 볼륨부터 소리로 보고, 그 40ms 전부터 남긴다
DEADLINE_S = float(os.getenv("DEADLINE_S", 18))           # 말 끝 이후 이 안에 마무리 (25초 안에 turn.end 필요)
MAX_TOOL_ROUNDS = 3                 # LLM이 함수를 연달아 부를 수 있는 최대 횟수
VERBOSE = os.getenv("VERBOSE", "1") == "1"                # 측정 때는 0으로 (고객 발화 내용을 출력하지 않음)

NOT_HEARD = "잘 못 들었습니다. 다시 한번 말씀해 주시겠어요?"
SORRY = "죄송합니다. 잠시 문제가 생겨 바로 답변드리기 어렵습니다."
# 자주 쓰는 말은 서버를 켤 때 미리 음성으로 만들어 둔다 (Gemini 가 목소리를 만드는 데 3~4초가 걸려서).
# LLM 이 아래 문장을 그대로 말하면 만드는 시간 없이 바로 나간다. 11/21 매뉴얼에 정해진 문구가 있으면 여기를 고칠 것.
PHRASES = {
    "주문번호나 전화번호가 필요한데 고객이 말하지 않았을 때": "주문번호를 알려 주시면 바로 확인해 드리겠습니다.",
    "건강 문제 등으로 상담원 연결이 필요할 때": "걱정되시겠어요. 바로 상담원 연결을 도와드리겠습니다.",
    "매뉴얼에 없는 내용을 물어서 답할 수 없을 때 (조회 결과가 '없음'인 경우는 해당하지 않으며, 그때는 '주문번호 ○○로는 조회되지 않습니다. 다시 확인해 주시겠어요?'처럼 짧게 알린다)":
        "죄송하지만 그 부분은 제가 정확히 안내드리기 어렵습니다. 상담원 연결을 도와드리겠습니다.",
}

PROMPT = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8")   # 매뉴얼은 이 파일에 붙여 넣는다
ai = AsyncOpenAI(timeout=10, max_retries=1)               # LLM 전용. 키는 환경변수 OPENAI_API_KEY
gemini = genai.Client(http_options=types.HttpOptions(      # 키는 GEMINI_API_KEY. preview 모델은 v1beta 로 부른다
    base_url=os.getenv("GEMINI_BASE_URL"), api_version="v1beta", timeout=10_000))
http = httpx.AsyncClient(timeout=10)
FIXED = {}                          # 고정 문구(사과 등) → 미리 만들어 둔 음성. warm_up() 이 채운다

# 서버를 켠 뒤 누적 사용량 (크레딧 관리용). 단가는 대략적인 추정이니 공식 가격 페이지로 확인할 것
USAGE = {"calls": 0, "llm_in": 0, "llm_out": 0, "stt_sec": 0.0, "tts_sec": 0.0}
PRICE = {"llm_in": 0.75e-6, "llm_out": 4.5e-6, "stt_sec": 0.12 / 3600, "tts_sec": 25 * 10e-6}   # 달러 (TTS 는 음성 1초=25토큰 가정)


def usage_line():
    cost = sum(USAGE[k] * PRICE[k] for k in PRICE)
    return (f"[사용량] 누적 {USAGE['calls']}통 | LLM {USAGE['llm_in']}/{USAGE['llm_out']} 토큰 | "
            f"STT {USAGE['stt_sec']:.0f}초 | TTS {USAGE['tts_sec']:.0f}초 | 추정 ${cost:.3f}")


def count_llm(resp):
    if resp.usage:
        USAGE["llm_in"] += resp.usage.prompt_tokens
        USAGE["llm_out"] += resp.usage.completion_tokens


class Lap:
    """단계별로 걸린 시간을 재서, 통화가 끝나면 한 줄로 출력한다."""
    def __init__(self, case_id):
        self.case_id, self.t, self.marks = case_id, time.perf_counter(), []

    def __call__(self, label):
        now = time.perf_counter()
        self.marks.append((label, now - self.t))
        self.t = now

    def report(self):
        steps = " | ".join(f"{label} {sec:.2f}s" for label, sec in self.marks)
        total = sum(sec for _, sec in self.marks)
        print(f"[{self.case_id}] 말 끝 감지 뒤: {steps} | 합계 {total:.2f}s", flush=True)


# ── 1. 받아쓰기: Soniox 실시간 (통화 한 통) ──────────────────────────
class Stt:
    """고객 목소리를 말하는 동안 Soniox 로 흘려보내고, 말이 끝나면 finalize 로 확정된 글자를 받는다."""

    def __init__(self):
        self.queue, self.text, self.error = asyncio.Queue(), "", None     # queue: bytes=목소리, str=제어 메시지
        self.ok, self.done = False, asyncio.Event()                      # ok: <fin> 을 받음 / done: 결과가 났거나 실패함
        self.partial = ""                                                # 아직 확정 전인 글자 (말 끝 판단 보조용)
        self.t0 = time.time()
        self.task = asyncio.create_task(self._run())

    def feed(self, pcm):
        self.queue.put_nowait(pcm)

    def looks_unfinished(self):
        """지금까지 받아쓴 글이 '~는데', '~하고' 처럼 말이 이어질 모양으로 끝났는가 (문장부호는 무시)."""
        tail = (self.text + self.partial).strip().rstrip(".,?!…~ ")
        return tail.endswith(CONTINUING)

    async def finish(self):
        self.queue.put_nowait(json.dumps({"type": "finalize"}))
        await asyncio.wait_for(self.done.wait(), STT_TIMEOUT_S)
        if not self.ok:
            raise RuntimeError(f"Soniox 받아쓰기 실패: {self.error or '연결이 끊김'}")
        return self.text.strip()

    async def close(self):                                   # 스트림 길이만큼 과금되니 통화가 끝나면 닫는다
        USAGE["stt_sec"] += time.time() - self.t0
        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task

    async def _run(self):
        try:
            async with websockets.connect(SONIOX_URL, additional_headers={"Authorization": f"Bearer {SONIOX_KEY}"}) as ws:
                await ws.send(json.dumps(SONIOX_CONFIG))
                sender = asyncio.create_task(self._send_loop(ws))
                try:
                    await self._receive_loop(ws)
                except asyncio.CancelledError:                # 통화가 끝나 닫는 경우: 정상 종료 코드로 연결을 닫는다
                    await ws.close()
                    raise
                finally:
                    sender.cancel()
        except Exception as e:
            self.error = self.error or repr(e)
        finally:
            self.done.set()

    async def _send_loop(self, ws):
        while True:
            await ws.send(await self.queue.get())

    async def _receive_loop(self, ws):
        async for message in ws:
            reply = json.loads(message)
            if reply.get("error_code"):
                self.error = f"{reply['error_code']} {reply.get('error_message', '')}"
                self.done.set()
                return
            self.partial = "".join(t["text"] for t in reply.get("tokens", []) if not t.get("is_final"))
            for token in reply.get("tokens", []):
                if not token.get("is_final"):
                    continue                                 # 임시 글자는 바뀔 수 있으니 확정된 것만 모은다
                if token["text"] == "<fin>":
                    self.ok = True
                    self.done.set()
                    return
                if token["text"] != "<end>":
                    self.text += token["text"]


# ── 2. 듣기: 말이 끝날 때까지 목소리를 Soniox 로 넘기며 기다린다 ──────
def rms(pcm):
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


async def listen(ws, stt):
    voiced_ms = silent_ms = total_ms = 0
    async for m in ws:
        if not isinstance(m, bytes):
            continue
        stt.feed(m)                                          # 받아쓰기는 고객이 말하는 동안 이미 진행된다
        ms = len(m) * 1000 // (RATE * 2)
        total_ms += ms
        if rms(m) >= SPEECH_RMS:
            voiced_ms, silent_ms = voiced_ms + ms, 0
        elif voiced_ms >= MIN_SPEECH_MS:                     # 말이 시작된 뒤의 조용한 시간만 센다
            silent_ms += ms
        wait_ms = LONG_SILENCE_MS if silent_ms >= END_SILENCE_MS and stt.looks_unfinished() else END_SILENCE_MS
        if silent_ms >= wait_ms or total_ms >= MAX_LISTEN_MS:
            break


async def drain(ws):
    """말이 끝난 뒤에도 측정기는 무음을 계속 보낸다. 읽어서 버린다."""
    with suppress(websockets.ConnectionClosed):
        async for _ in ws:
            pass


# ── 3. 생각: LLM + 모의 함수 서버 ───────────────────────────────────
async def load_tools(start):
    """함수 목록(GET /tools)은 OpenAI 도구 형식 그대로라서 LLM에 바로 넘긴다."""
    try:
        r = await http.get(f"{start['tools_url'].rstrip('/')}/tools")
        r.raise_for_status()
        return r.json()
    except (httpx.HTTPError, ValueError, KeyError, AttributeError) as e:
        print(f"[{start.get('case_id')}] 함수 목록을 못 가져옴: {e!r}", flush=True)
        return []


def check_args(name, args, tools):
    """문지기: 함수를 부르기 전에 검사한다. 문제가 있으면 이유(문자열)를, 괜찮으면 None을 돌려준다."""
    spec = next((t["function"] for t in tools if t["function"]["name"] == name), None)
    if spec is None:
        return f"정의에 없는 함수입니다: {name}"
    if not isinstance(args, dict):
        return "arguments는 JSON 객체여야 합니다"
    missing = [k for k in spec.get("parameters", {}).get("required", []) if not str(args.get(k, "")).strip()]
    if missing:
        return f"필요한 값이 없습니다: {', '.join(missing)}. 고객에게 먼저 물어보세요."
    return None


async def run_tool(start, tools, call):
    name = call.function.name
    try:
        args = json.loads(call.function.arguments or "{}")
    except ValueError:
        return {"error": "arguments가 올바른 JSON이 아닙니다"}
    if problem := check_args(name, args, tools):
        return {"error": problem}
    try:                                                  # X-Session-Id 가 없으면 호출하지 않은 것으로 채점된다
        r = await http.post(f"{start['tools_url'].rstrip('/')}/tools/{name}", json=args,
                            headers={"X-Session-Id": start["session_id"]})
        return r.json()
    except (httpx.HTTPError, ValueError) as e:
        return {"error": f"함수 호출 실패: {e!r}"}


def history_messages(history):
    """이전 대화를 LLM 메시지로 바꾼다. 형식은 11/21 규격에 맞춰 고칠 것 (지금은 role/content 또는 speaker/text 가정)."""
    roles = {"customer": "user", "user": "user", "agent": "assistant", "assistant": "assistant"}
    out = []
    for h in history or []:
        if isinstance(h, dict):
            role = roles.get(str(h.get("role") or h.get("speaker") or "").lower())
            text = h.get("content") or h.get("text")
            if role and text:
                out.append({"role": role, "content": str(text)})
    return out


def system_prompt(start):
    info = f"\n\n[통화 정보]\n현재 시각: {start.get('now')}"
    if start.get("caller"):
        info += f"\n발신자: {json.dumps(start['caller'], ensure_ascii=False)}"
    history = start.get("history")
    if history and not history_messages(history):             # 형식을 못 알아봤어도 맥락이 사라지지 않게 원문을 그대로 보여 준다
        info += f"\n이전 대화(원문): {json.dumps(history, ensure_ascii=False)}"
    fixed = "\n\n[고정 문장]\n아래 상황에서는 해당 문장을 한 글자도 바꾸지 말고 그 문장만 말한다.\n" + "\n".join(
        f"- {when}: {line}" for when, line in PHRASES.items())
    return PROMPT + fixed + info


def llm_options(tools=None):
    options = {"tools": tools} if tools else {}
    if LLM_EFFORT:
        options["reasoning_effort"] = LLM_EFFORT
    return options


async def think(start, text, tools):
    messages = [{"role": "system", "content": system_prompt(start)},
                *history_messages(start.get("history")),
                {"role": "user", "content": text}]
    for _ in range(MAX_TOOL_ROUNDS + 1):
        resp = await ai.chat.completions.create(model=LLM_MODEL, messages=messages, **llm_options(tools))
        count_llm(resp)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return (msg.content or "").strip() or SORRY
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in msg.tool_calls]})
        for call in msg.tool_calls:
            result = await run_tool(start, tools, call)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
    return SORRY


# ── 4. 말하기: Gemini TTS ───────────────────────────────────────────
def sentences(text):
    """TTS 로 읽힐 단위로 나눈다. TTS_PER_SENTENCE=0 이면 답 전체를 한 번에 읽혀 요청 수를 줄인다."""
    if not TTS_PER_SENTENCE:
        return [text.strip()] if text.strip() else []
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def tts_audio(r):
    """Gemini 응답에서 음성 데이터를 찾는다. 없으면 None."""
    for cand in r.candidates or []:
        for part in (cand.content.parts if cand.content else None) or []:
            if part.inline_data and part.inline_data.data:
                return part.inline_data.data
    return None


def tts_problem(r):
    """음성이 없을 때 이유(종료 사유, 차단 사유, 글자로만 답했는지)를 한 줄로 정리한다."""
    cand = r.candidates[0] if r.candidates else None
    parts = (cand.content.parts if cand and cand.content else None) or []
    said = " ".join(p.text for p in parts if p.text)[:60]
    block = r.prompt_feedback.block_reason if r.prompt_feedback else None
    return f"종료사유={cand.finish_reason if cand else None}, 차단사유={block}, 글자 응답={said!r}, 모델={r.model_version}"


def tts_config():
    return types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE))))


async def gemini_tts(text):
    """Gemini TTS 를 한 번 부른다. (24kHz PCM 음성, 실패 이유)를 돌려준다. 음성이 없으면 음성 자리는 None."""
    r = await gemini.aio.models.generate_content(model=TTS_MODEL, contents=text, config=tts_config())
    audio = tts_audio(r)
    return audio, ("" if audio else tts_problem(r))


def to_16k(raw):
    """Gemini 의 24kHz PCM 을 대회 규격 16kHz 로 바꾼다."""
    raw = raw[: len(raw) // 2 * 2]
    g = math.gcd(RATE, TTS_RATE)
    pcm = resample_poly(np.frombuffer(raw, dtype=np.int16), RATE // g, TTS_RATE // g)
    return np.clip(pcm, -32768, 32767).astype(np.int16).tobytes()


async def synth(sentence):
    """문장 하나를 Gemini TTS 로 읽혀 16kHz PCM 으로 만든다."""
    problem = ""
    for style in (TTS_STYLE, TTS_KO_STYLE, TTS_RETRY_STYLE):  # 실패하면 읽어 달라는 지시를 바꿔 가며 최대 3번 시도한다
        try:
            raw, problem = await gemini_tts(style + sentence)
        except genai_errors.ClientError as e:
            if "generate text" not in str(e):                 # 글로 답하려다 실패한 경우만 다시 시도하고, 키·한도 같은 문제는 그대로 알린다
                raise
            raw, problem = None, str(e)[:150]
        if raw:
            break
    else:
        raise RuntimeError(f"Gemini TTS 가 음성을 돌려주지 않았습니다 ({problem})")
    pcm = to_16k(raw)
    USAGE["tts_sec"] += len(pcm) / (RATE * 2)
    return pcm


def lead_start(pcm):
    """앞 무음이 끝나는 위치(바이트). 소리가 없으면 None."""
    for i in range(0, len(pcm) - FRAME + 1, FRAME):
        if rms(pcm[i:i + FRAME]) >= LEAD_RMS:
            return max(0, i - LEAD_KEEP_MS * RATE * 2 // 1000)
    return None


def trim_lead(pcm):
    """음성 앞쪽의 무음을 잘라낸다. 측정기는 소리가 처음 도착한 시각을 '첫 소리'로 본다."""
    start = lead_start(pcm)
    return pcm if start is None else pcm[start:]


def phrase_key(text):
    """고정 문구를 찾을 때 띄어쓰기와 문장부호 차이는 무시한다."""
    return re.sub(r"[\s.,!?~…]", "", text)


async def stream_tts(text):
    """Gemini 가 음성을 조각으로 보내는 대로 16kHz PCM 조각을 내보낸다 (24kHz→16kHz 는 샘플 3개 단위로 끊어서 변환)."""
    stream = await gemini.aio.models.generate_content_stream(model=TTS_MODEL, contents=TTS_STYLE + text, config=tts_config())
    carry, lead = b"", True
    async for chunk in stream:
        raw = tts_audio(chunk)
        if not raw:
            continue
        carry += raw
        cut = len(carry) // 6 * 6
        if not cut:
            continue
        piece, carry = to_16k(carry[:cut]), carry[cut:]
        if lead:                                              # 맨 앞 무음은 건너뛴다
            start = lead_start(piece)
            if start is None:
                continue
            piece, lead = piece[start:], False
        USAGE["tts_sec"] += len(piece) / (RATE * 2)
        yield piece


async def speak(sentence):
    """문장 하나를 통째로 읽혀서(앞 무음은 자르고) 돌려준다. 고정 문구는 미리 만들어 둔 것을 쓴다."""
    return FIXED.get(phrase_key(sentence)) or trim_lead(await synth(sentence))


async def speak_chunks(text):
    """문장 하나를 읽혀서 16kHz PCM 조각을 만들어지는 대로 내보낸다. 고정 문구는 미리 만든 것을, 스트리밍이 안 되면 통째로 만든 것을 쓴다."""
    ready = FIXED.get(phrase_key(text))
    if ready:
        yield ready
        return
    if TTS_STREAM:
        sent = False
        try:
            async for piece in stream_tts(text):
                sent = True
                yield piece
        except Exception as e:
            if sent or "RESOURCE_EXHAUSTED" in str(e):        # 이미 말하기 시작했거나 한도 초과면 되돌리지 않고 그대로 알린다
                raise
            print(f"스트리밍 실패, 통째 방식으로 다시 시도: {str(e)[:80]}", flush=True)
        else:
            if sent:
                return
    yield trim_lead(await synth(text))


async def say(ws, text, lap=None):
    """답을 읽혀서, 음성이 만들어지는 대로 20ms 조각으로 바로 내보낸다."""
    await ws.send(json.dumps({"type": "agent.transcript", "text": text}, ensure_ascii=False))
    pending, first = b"", True
    for unit in sentences(text):
        async for piece in speak_chunks(unit):
            pending += piece
            while len(pending) >= FRAME:
                await ws.send(pending[:FRAME])
                pending = pending[FRAME:]
                if first and lap:
                    lap("첫 소리 전송")
                    first = False
    if pending:
        await ws.send(pending.ljust(FRAME, b"\0"))


# ── 5. 한 통화 처리 ─────────────────────────────────────────────────
async def respond(ws, start, stt, tools, lap):
    cid = start.get("case_id")
    text = await stt.finish()
    lap("받아쓰기 확정")
    reply = await think(start, text, await tools) if text else NOT_HEARD
    lap("생각")
    if VERBOSE:
        print(f"[{cid}] 고객: {text}\n[{cid}] 상담원: {reply}", flush=True)
    await say(ws, reply, lap)
    lap("말하기 끝")


async def talk(ws, start, stt):
    tools = asyncio.create_task(load_tools(start))            # 고객이 말하는 동안 미리 받아 둔다
    await listen(ws, stt)
    lap, sink = Lap(start.get("case_id")), asyncio.create_task(drain(ws))
    try:
        await asyncio.wait_for(respond(ws, start, stt, tools, lap), DEADLINE_S)
    except Exception as e:                                    # 무슨 일이 있어도 끝 신호는 보낸다
        print(f"[{start.get('case_id')}] 오류: {e!r}", flush=True)
        lap("실패")
        with suppress(Exception):
            await asyncio.wait_for(say(ws, SORRY), 4)
    finally:
        sink.cancel()
        lap.report()
        USAGE["calls"] += 1
        print(usage_line(), flush=True)
        await ws.send(json.dumps({"type": "turn.end"}))


async def handle(ws):
    with suppress(websockets.ConnectionClosed):
        start = json.loads(await ws.recv())                   # session.start
        stt = Stt()                                           # 고객이 말하는 동안 받아쓰기 연결을 미리 열어 둔다
        try:
            await talk(ws, start, stt)
        finally:
            await stt.close()


async def warm_up():
    """첫 통화만 느려지지 않게, 서버를 켤 때 자주 쓰는 문구를 음성으로 만들어 두고(사과 음성은 API 가 실패해도 나간다) LLM 연결도 데워 둔다."""
    async def fill(unit):
        with suppress(Exception):
            FIXED[phrase_key(unit)] = trim_lead(await synth(unit))

    async def llm():
        with suppress(Exception):
            count_llm(await ai.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": "안녕하세요"}]))

    lines = [NOT_HEARD, SORRY, *PHRASES.values()]
    await asyncio.gather(llm(), *(fill(unit) for line in lines for unit in sentences(line)))
    print(f"[준비 완료] 미리 만든 고정 문구 음성 {len(FIXED)}개. 이제 테스트를 시작해도 돼요.", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    port = ap.parse_args().port
    async with websockets.serve(handle, "0.0.0.0", port, max_size=None):
        print(f"AI 상담원 서버 시작: ws://0.0.0.0:{port}  (LLM={LLM_MODEL}, 스트리밍={'켬' if TTS_STREAM else '끔'}) - '[준비 완료]'가 나올 때까지 기다리세요", flush=True)
        overrides = {k: os.environ[k] for k in ("OPENAI_BASE_URL", "GEMINI_BASE_URL", "SONIOX_URL") if os.getenv(k)}
        if overrides:
            print(f"[주의] 기본 주소 대신 다른 주소를 쓰는 중입니다 (시험용 가짜 서버일 수 있음): {overrides}", flush=True)
        warm = asyncio.create_task(warm_up())
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
