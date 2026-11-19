# [API 점검기] 실제 키로 STT · LLM · TTS 를 차례로 한 번씩만 불러 보고, 잘 되는지와 걸린 시간을 알려 줍니다.
# 왜 있나: 상담원 전체를 돌리기 전에 어느 서비스가 문제인지 크레딧을 거의 안 쓰고 확인하려고요 (제출물에서는 빼도 돼요).
"""사용:  python agent_server/check_apis.py                 # 내가 녹음한 한국어 wav(data/audio/ko)가 있으면 그걸, 없으면 스타터의 영어 데모로 STT 시험
        python agent_server/check_apis.py --audio 내음성.wav   # STT 에 쓸 wav 를 직접 지정 (16kHz 모노 16-bit)
  1) STT  wav 를 Soniox 에 실시간으로 입력해 받아쓰기      2) LLM  짧은 인사      3) LLM  함수 호출이 되는지
  4) TTS  한국어 문장을 읽혀 wav 로 저장 (직접 들어 보세요)    5) STT  4)의 음성을 다시 받아쓰기 → 한국어 숫자가 어떻게 적히는지
각 단계는 앞 단계가 실패해도 계속 진행합니다.
"""
import argparse, asyncio, sys, time, wave
from pathlib import Path

import server                                   # .env 읽기와 키 검사도 server.py 가 한다

ROOT = Path(__file__).resolve().parents[1]
SENTENCE = "주문번호는 일이삼사입니다."                      # TTS → STT 왕복 시험용: Gemini 가 안정적으로 읽는 상담원 투의 문장
CUSTOMER = "제 전화번호는 공일공 일이삼사 오육칠팔이에요."      # 고객 투의 긴 번호 문장: Gemini 가 음성 없이 끝내는 경우가 있어 진단에만 쓴다
PHONE = "전화번호는 공일공 일이삼사 오육칠팔입니다."           # 상담원이 번호를 읽어 주는 문장도 되는지 참고용으로 한 번 시험한다
TOOL = {"type": "function", "function": {
    "name": "get_order_info", "description": "주문 번호로 배송 상태를 조회한다",
    "parameters": {"type": "object", "required": ["order_id"], "properties": {"order_id": {"type": "string"}}}}}
# TTS 가 실패할 때 원인을 좁히기 위한 시험: (이름, 읽어 달라는 지시) × (문장 종류)
STYLES = [("Say", "Say: "), ("한글지시", "다음 문장을 그대로 읽어 주세요: "), ("ReadAloud", "Read aloud: ")]
PROBES = {"숫자 포함 고객 문장": CUSTOMER, "숫자 포함 상담원 문장": SENTENCE, "숫자 없는 상담원 문장": "배송 중입니다."}
FAILED = []


async def step(name, coro):
    print(f"\n[{name}]", flush=True)
    t = time.perf_counter()
    try:
        result = await coro
    except Exception as e:
        FAILED.append(name)
        print(f"   결과: 실패 - {e!r}", flush=True)
        return None
    print(f"   결과: 성공 ({time.perf_counter() - t:.2f}초)", flush=True)
    return result


def save_wav(path, pcm):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(server.RATE)
        w.writeframes(pcm)


def pick_audio(arg):
    """STT 시험에 쓸 wav: 직접 지정 > 내가 녹음한 한국어(data/audio/ko) > 스타터의 영어 데모."""
    if arg:
        return Path(arg)
    mine = sorted((ROOT / "data" / "audio" / "ko").glob("*.wav"))
    return mine[0] if mine else ROOT / "data" / "audio" / "demo" / "DEMO1.wav"


# ── STT ─────────────────────────────────────────────────────────────
async def transcribe(pcm):
    """목소리를 실시간 속도로 Soniox 에 흘려보내고 (받아쓴 글, finalize 요청 뒤 확정까지 걸린 시간)을 돌려준다."""
    s, silence = server.Stt(), b"\0" * server.FRAME
    speech = [pcm[i:i + server.FRAME].ljust(server.FRAME, b"\0") for i in range(0, len(pcm), server.FRAME)]
    for frame in [silence] * 15 + speech + [silence] * 40:            # 앞 0.3초 무음 + 음성 + 뒤 0.8초 무음, 실시간 속도로
        s.feed(frame)
        await asyncio.sleep(0.02)
    t = time.perf_counter()
    try:
        text = await s.finish()
    finally:
        await s.close()
    return text, time.perf_counter() - t


async def stt_file(path):
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (server.RATE, 1, 2):
            raise ValueError("16kHz / 모노 / 16-bit wav 가 아닙니다")
        pcm = w.readframes(w.getnframes())
    print(f"   입력 음성: {path.name} ({len(pcm) / (server.RATE * 2):.1f}초, 실시간 속도로 입력하니 그만큼 걸려요)")
    text, lag = await transcribe(pcm)
    print(f"   받아쓴 글: {text!r}   (finalize 요청 뒤 확정까지 {lag:.2f}초)")
    if not text:
        print("   ! 글자가 비어 있어요. 음성에 말소리가 있는지 확인하세요.")
    elif "demo" in path.parent.name:
        print("   → 스타터의 영어 데모 음성이라 영어로 받아쓰는 게 정상이에요. 한국어 시험은 eval/record.py 로 녹음한 뒤 다시 돌려 보세요.")


async def stt_roundtrip(pcm):
    text, lag = await transcribe(pcm)
    print(f"   받아쓴 글: {text!r}   (finalize 요청 뒤 확정까지 {lag:.2f}초)")
    print("   원래 문장: " + repr(SENTENCE))
    print("   → 숫자가 아라비아 숫자로 적혔습니다." if any(c.isdigit() for c in text)
          else "   → 숫자가 한글 그대로 적혔습니다. LLM이 숫자로 바꾸도록 prompt.md 규칙이 꼭 필요합니다.")


# ── LLM ─────────────────────────────────────────────────────────────
async def llm_hello():
    r = await server.ai.chat.completions.create(
        model=server.LLM_MODEL, messages=[{"role": "user", "content": "한 문장으로 짧게 인사해 주세요."}], **server.llm_options())
    server.count_llm(r)
    print(f"   답: {r.choices[0].message.content!r}  (모델 {r.model})")


async def llm_tool():
    r = await server.ai.chat.completions.create(
        model=server.LLM_MODEL, **server.llm_options([TOOL]),
        messages=[{"role": "system", "content": server.PROMPT}, {"role": "user", "content": "주문번호 1234번 배송 조회해 주세요."}])
    server.count_llm(r)
    calls = r.choices[0].message.tool_calls
    if calls:
        print("   함수 호출:", [(c.function.name, c.function.arguments) for c in calls])
    else:
        print(f"   함수를 부르지 않고 말로만 답했습니다: {r.choices[0].message.content!r}  ← 프롬프트나 모델 설정 확인 필요")


# ── TTS ─────────────────────────────────────────────────────────────
async def diagnose_tts(out):
    """TTS 가 실패했을 때 원인을 좁힌다: 지시 문구 × 문장 종류별로 되는지 보여 주고, 성공한 소리는 저장한다."""
    print("   [진단] 지시 문구와 문장 종류를 바꿔 가며 시험합니다 (9번 호출)")
    for name, text in PROBES.items():
        results = []
        for label, style in STYLES:
            try:
                audio, problem = await server.gemini_tts(style + text)
                results.append(f"{label}={'성공' if audio else '음성없음'}")
                if audio and text == PROBES["숫자 없는 상담원 문장"]:
                    save_wav(out / f"diag_{label}.wav", server.to_16k(audio))     # 지시 문구를 소리로 읽는지 직접 들어 보라고 저장
            except Exception as e:
                results.append(f"{label}=오류({str(e)[:45]})")
        print(f"   [진단] {name}: " + ", ".join(results))
    print(f"   [진단] 성공한 소리는 {out} 의 diag_*.wav 에 저장했어요. 지시 문구를 소리로 읽어 버리지 않는지 들어 보세요.")
    try:
        names = [m.name async for m in await server.gemini.aio.models.list() if "tts" in (m.name or "")]
        print(f"   [진단] 이 키로 쓸 수 있는 TTS 모델: {names or '없음'}  (지금 쓰는 모델: {server.TTS_MODEL})")
    except Exception as e:
        print(f"   [진단] 모델 목록을 못 가져왔어요: {e!r}")


async def phone_probe():
    """참고 시험(실패해도 문제로 치지 않음): 상담원이 전화번호를 읽어 주는 문장도 되는지."""
    try:
        audio, problem = await server.gemini_tts(server.TTS_STYLE + PHONE)
        print("   전화번호 문장도 읽습니다." if audio else f"   전화번호 문장은 음성이 비어 옵니다 ({problem}) → 번호를 읽어 주는 답은 피하는 게 좋아요.")
    except Exception as e:
        print(f"   참고 시험 중 오류: {str(e)[:100]}")


async def stream_probe():
    """참고 시험(실패해도 문제로 치지 않음): 음성을 조각으로 나눠 받으면 첫 소리가 얼마나 빨리 오는지. 4번의 걸린 시간과 비교해 보세요."""
    t0, first, chunks, size = time.perf_counter(), None, 0, 0
    try:
        stream = await server.gemini.aio.models.generate_content_stream(
            model=server.TTS_MODEL, contents=server.TTS_STYLE + SENTENCE, config=server.tts_config())
        async for chunk in stream:
            audio = server.tts_audio(chunk)
            if audio:
                chunks, size = chunks + 1, size + len(audio)
                first = first or time.perf_counter() - t0
    except Exception as e:
        print(f"   스트리밍 시험 중 오류: {str(e)[:120]}")
        return
    if not chunks:
        print("   스트리밍으로는 음성 조각이 오지 않았어요 (지원하지 않을 수 있어요).")
    else:
        print(f"   스트리밍: 첫 조각까지 {first:.2f}초, 전체 {time.perf_counter() - t0:.2f}초, 조각 {chunks}개 (음성 {size / 48000:.1f}초 분량)")
        print("   → 첫 조각까지의 시간이 4번의 걸린 시간보다 훨씬 짧으면, 말하기를 스트리밍으로 바꿔서 첫 소리를 크게 앞당길 수 있어요.")


async def tts(out):
    try:
        raw = await server.synth(SENTENCE)
    except Exception:
        await diagnose_tts(out)
        raise
    levels = [server.rms(raw[i:i + server.FRAME]) for i in range(0, len(raw) - server.FRAME + 1, server.FRAME)]
    lead_ms = next((i for i, v in enumerate(levels) if v >= server.LEAD_RMS), len(levels)) * 20
    pcm = server.trim_lead(raw)
    path = out / "tts_sample.wav"
    save_wav(path, pcm)
    print(f"   음성 길이 {len(raw) / (server.RATE * 2):.1f}초, 앞 무음 약 {lead_ms}ms (상담원 서버는 이 부분을 잘라서 보냄)")
    print(f"   저장: {path}")
    print("   → 직접 재생해서 속도와 높낮이가 자연스러운지, 읽어 달라는 지시 문구를 소리로 읽지 않는지 들어 보세요.")
    print("     느리거나 낮게 들리면 server.py 의 TTS_RATE 를 확인하세요.")
    return pcm


async def main(out, audio):
    out.mkdir(parents=True, exist_ok=True)
    await step("1. STT 받아쓰기 (wav 를 Soniox 에 입력)", stt_file(audio))
    await step("2. LLM 인사", llm_hello())
    await step("3. LLM 함수 호출", llm_tool())
    pcm = await step("4. TTS 한국어 음성", tts(out))
    if pcm:
        await step("4-1. 참고: 전화번호를 읽어 주는 문장", phone_probe())
        await step("4-2. 참고: 스트리밍(조각으로 받기) 시험", stream_probe())
        await step("5. STT 왕복 (4번 음성을 다시 받아쓰기: 한국어 숫자 표기 확인)", stt_roundtrip(pcm))
    print("\n" + server.usage_line())
    print("모든 단계가 정상입니다." if not FAILED else f"문제가 있는 단계: {', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="1번 STT 시험에 쓸 wav (16kHz 모노 16-bit). 없으면 내 녹음 또는 스타터 영어 데모")
    ap.add_argument("--out", default=str(ROOT / "runs" / "check_apis"), help="음성 샘플을 저장할 폴더")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(Path(args.out), pick_audio(args.audio))))
