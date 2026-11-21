# [말하기 방식 비교] 같은 상담원 문장을 TTS 설정별로 읽혀서 첫 음성까지·전부 준비까지 걸린 시간과 재생 끊김을 잽니다.
# 왜 있나: 문장별 동시 요청, 키 여러 개, 같은 글 이중 요청 중 무엇이 첫 소리를 당기는지 같은 문장으로 공정하게 비교하려고요.
# 비용: Gemini TTS 요청 = 방식별 조각 수의 합. 시작 전에 요청 수를 보여 줍니다.
"""사용 (저장소 맨 위에서):
  python eval/bench_tts.py --dry-run                 # 요청 수만 보기 (API 호출 없음)
  python eval/bench_tts.py --out runs/tts2           # 기본: 문장 3개 × 설정 6개 × 2번
  python eval/bench_tts.py --out runs/tts3 --presets stream,split_k2,hedge2 --reps 3
  python eval/bench_tts.py --out runs/tts4 --presets stream,split3_k3,split3any_k3 --texts T4,T5,T3   # 조회 뒤 답을 3조각까지
설정 (eval/tts_presets.py, 서버의 같은 환경변수로 그대로 켤 수 있음)
  stream           한 번에 (지금 서버)
  split_k1         문장별 동시 요청, 키 1개
  split_k2         문장별 동시 요청, 키 2개에 나눠 (GEMINI_API_KEY2 필요)
  headtail_k2      앞 문장 / 나머지 2조각만, 키 2개에 나눠
  hedge2           안 쪼개고 같은 글을 키 2개로 동시에 요청해 먼저 온 것 사용
  headtail_hedge2  2조각 + 조각마다 키 2개로 동시 요청
  split3_k3        최대 3조각 (문장 + 쉼표·연결 어미 뒤에서도 끊음), 조각마다 다른 키 (GEMINI_API_KEY3 까지)
  split3any_k3     최대 3조각 (+ '은·는·로는' 같은 주제 조사 뒤에서도 끊음), 조각마다 다른 키
  문장별 설정은 '모두 받은 뒤 합쳐 보내기'(parallel) 시간도 같은 요청으로 함께 계산합니다.
  같은 문장 음성을 다시 쓰는 캐시는 끄고 잽니다 (매번 실제로 만든 시간).
결과: <out>/tts_bench.jsonl, <out>/summary.md, <out>/wav/*.wav (이어 붙인 결과를 직접 들어 보세요)
"""
import argparse, asyncio, json, statistics, sys, time, wave
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent_server"))
import server                                                 # .env 읽기와 키 검사, TTS 코드는 server.py 를 그대로 쓴다
sys.path.insert(0, str(ROOT / "eval"))
from tts_presets import PRESETS, requests_per_answer

TEXTS = {
    "T1": "안녕하세요, 다이렉트온 상담센터입니다. 곧 요청하신 작업을 진행해 드릴게요.",
    "T2": "주문번호 1234 상품은 현재 배송 중입니다. 내일 중으로 도착할 예정입니다.",
    "T3": "확인해 보니 주문번호 1243 상품은 오늘 오전에 출고되었습니다. 택배사 사정에 따라 하루 이틀 걸릴 수 있습니다. 더 궁금하신 점 있으실까요?",
    "T4": "주문번호 1234로는 조회되지 않습니다. 다시 확인해 주시겠어요?",                               # 조회 뒤 답 (실험 KO01·KO07 실제 답)
    "T5": "주문번호 2002 상품은 배송이 완료되었고 어제 오후에 문 앞에 놓였다고 기록되어 있습니다.",       # 조회 뒤 답, 한 문장
}


def apply(name):
    """설정을 server 모듈에 바로 적용한다 (서버를 다시 켜지 않고 같은 코드로 잰다)."""
    env = PRESETS[name][1]
    server.TTS_KEYS = int(env.get("TTS_KEYS", 0))
    server.TTS_HEDGE = int(env.get("TTS_HEDGE", 1))
    server.TTS_MAX_PARTS = int(env.get("TTS_MAX_PARTS", 3))
    server.TTS_SPLIT_CLAUSE = int(env.get("TTS_SPLIT_CLAUSE", 0))
    return env["TTS_MODE"]


def requests_for(text, name):
    apply(name)
    return requests_per_answer(name, len(server.split_sentences(text)))


class Pacer:
    """--rpm 을 주면 최근 60초 요청 수가 그 이하가 되게 기다린다 (기본 0 = 기다리지 않음)."""

    def __init__(self, rpm):
        self.rpm, self.times = rpm, deque()

    async def wait(self, n):
        while self.rpm > 0:
            now = time.time()
            while self.times and now - self.times[0] > 60:
                self.times.popleft()
            if len(self.times) + n <= self.rpm:
                self.times.extend([now] * n)
                return
            wait = 60 - (now - self.times[0]) + 0.5
            print(f"   (--rpm {self.rpm} 에 맞춰 {wait:.0f}초 쉽니다)", flush=True)
            await asyncio.sleep(wait)


def save_wav(path, pcm):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(server.RATE)
        w.writeframes(pcm)


async def run_one(text_id, text, name, rep, out):
    mode = apply(name)
    trace = server.TtsTrace(mode)
    pcm, error = b"", None
    try:
        async for piece in server.tts_pieces(text, mode, trace):
            pcm += piece
    except Exception as e:
        error = repr(e)[:200]
    s = trace.summary()
    rows = [{"text_id": text_id, "preset": name, "rep": rep, "error": error, **s}]
    if mode == "parallel_ordered" and not error:                 # 같은 요청으로 '모두 받은 뒤 합쳐 보내기'도 계산한다
        rows.append({**rows[0], "preset": name + "+합쳐보냄", "derived_from": name,
                     "first_audio_ms": s["all_ready_ms"], "stall_ms": 0.0})
    if pcm:
        save_wav(out / "wav" / f"{text_id}_{name}_r{rep}.wav", pcm)
    return rows


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs)) if xs else None


def summarize(rows, out):
    names = list(dict.fromkeys(r["preset"] for r in rows))
    base = {(r["text_id"], r["rep"]): r for r in rows if r["preset"] == "stream" and not r["error"]}
    lines = ["| 설정 | 실행 | 첫 음성 중앙값(ms) | 최대(ms) | 전부 준비 중앙값(ms) | 재생 끊김 중앙값(ms) | 한 번에 대비 (같은 문장·회차 짝, 중앙값) | 실패 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        rs = [r for r in rows if r["preset"] == n]
        ok = [r for r in rs if not r["error"]]
        diff = [r["first_audio_ms"] - base[(r["text_id"], r["rep"])]["first_audio_ms"] for r in ok
                if (r["text_id"], r["rep"]) in base and r["first_audio_ms"] is not None]
        label = PRESETS.get(n.split("+")[0], (n,))[0] + (" → 모두 받은 뒤 합쳐 보냄" if "+" in n else "")
        lines.append(f"| {n} ({label}) | {len(rs)} | {med([r['first_audio_ms'] for r in ok])} | "
                     f"{round(max(r['first_audio_ms'] for r in ok)) if ok else None} | {med([r['all_ready_ms'] for r in ok])} | "
                     f"{med([r['stall_ms'] for r in ok])} | {(f'{med(diff):+d}ms ({sum(d < 0 for d in diff)}/{len(diff)} 빨라짐)') if diff and n != 'stream' else '-'} | "
                     f"{len(rs) - len(ok)} |")
    lines += ["", "| 문장 | 설정 | 회차 | 조각 | 첫 음성(ms) | 전부 준비(ms) | 끊김(ms) | 조각별 [글자, 완료 ms, 키, 먼저 온 요청] |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        if r.get("derived_from"):
            continue
        segs = " ".join(f"[{x.get('chars')}자,{x.get('done_ms')},키{x.get('keys')},{x.get('winner')}]" for x in r["segments"])
        lines.append(f"| {r['text_id']} | {r['preset']} | {r['rep']} | {r['parts']} | {r['first_audio_ms']} | {r['all_ready_ms']} | "
                     f"{r['stall_ms']} | {segs}{' 오류 ' + r['error'][:80] if r['error'] else ''} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[: len(names) + 2]), flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/tts_bench")
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--texts", default="T1,T2,T3", help="쓸 문장 id (T1~T5)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--rpm", type=int, default=0, help="분당 최대 TTS 요청 수 (0 = 제한 없음)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    names = [n for n in a.presets.split(",") if n]
    bad = [n for n in names if n not in PRESETS]
    if bad:
        raise SystemExit(f"모르는 설정: {bad} (있는 것: {', '.join(PRESETS)})")
    server.TTS_CACHE_ALL = False                                  # 같은 문장을 다시 쓰지 않고 매번 실제로 만든다
    texts = {k: TEXTS[k] for k in a.texts.split(",")}
    plan = [(tid, n, rep) for rep in range(1, a.reps + 1) for i, tid in enumerate(texts)
            for n in (names[(i + rep) % len(names):] + names[: (i + rep) % len(names)])]   # 문장·회차마다 순서를 돌려 시간대 영향을 고르게 한다
    total = sum(requests_for(texts[t], n) for t, n, _ in plan)
    print(f"계획: 실행 {len(plan)}번, Gemini TTS 요청 약 {total}회, Gemini 키 {len(server.TTS_CLIENTS)}개 찾음", flush=True)
    if len(server.TTS_CLIENTS) < 2 and any(int(PRESETS[n][1].get("TTS_KEYS", 1)) > 1 for n in names):
        print("   (키가 1개뿐이라 '키 2개' 설정도 같은 키로 보냅니다. .env 에 GEMINI_API_KEY2 를 넣어 주세요)", flush=True)
    for n in names:                                               # 설정마다 어떻게 나뉘는지 보여 준다
        mode = apply(n)
        if mode in ("stream", "single"):
            continue
        for tid, text in texts.items():
            print(f"  [{n}] {tid}: {server.split_sentences(text)}", flush=True)
    if a.dry_run:
        return
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    pacer, rows = Pacer(a.rpm), []
    for c in server.TTS_CLIENTS:                                  # TTS 요청이 아닌 모델 목록 조회로 키마다 연결(TLS)을 먼저 열어 둔다
        try:
            _ = [m async for m in await c.aio.models.list()][:1]
        except Exception as e:
            print(f"   (연결 예열 실패, 계속 진행: {e!r})", flush=True)
    with open(out / "tts_bench.jsonl", "a", encoding="utf-8") as f:
        for tid, n, rep in plan:
            await pacer.wait(requests_for(texts[tid], n))
            for row in await run_one(tid, texts[tid], n, rep, out):
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                err = f" 오류 {row['error'][:80]}" if row["error"] else ""
                print(f"{tid} r{rep} {row['preset']:<22} 첫 음성 {row['first_audio_ms']}ms · 전부 준비 {row['all_ready_ms']}ms · "
                      f"끊김 {row['stall_ms']}ms{err}", flush=True)
            await asyncio.sleep(1)
    summarize(rows, out)
    print(f"\n{server.usage_line()}\n결과: {out / 'summary.md'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
