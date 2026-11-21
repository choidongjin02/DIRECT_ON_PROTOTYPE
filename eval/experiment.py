# [실제 음성 실험 한 번에] 전산실(8100)과 상담원 서버를 말하기 방식별로 켜고, 녹음한 손님 음성을 한 통씩 걸어 비교합니다.
# 왜 있나: 터미널 3개를 띄우고 방식마다 서버를 다시 켜는 번거로움 없이, 같은 녹음으로 방식별 첫 소리를 비교하려고요.
# 비용: 통화마다 Soniox(음성 길이만큼) · gpt-5.4-mini · Jev · Gemini TTS(방식에 따라 1~3회)를 실제로 씁니다.
"""사용 (저장소 맨 위에서, PowerShell 이면 앞에 $env:PYTHONUTF8="1"; 를 붙여도 됨):
  python eval/experiment.py --out runs/exp1        # 기본: KO01·KO04·KO08 × 말하기 3가지(stream, split_k2, headtail_hedge2) × 업무 분류 jev
  python eval/experiment.py --out runs/exp2 --tts stream --routers off,jev          # 업무 분류 끔/켬만 비교
  python eval/experiment.py --out runs/exp3 --tts stream,headtail_hedge2+short       # +short = LLM 에 '첫 문장 짧게' 규칙 추가
  python eval/experiment.py --out runs/exp4 --tts stream,stream+early                # +early = 말 끝 판정을 기다리는 동안 LLM·TTS 미리 시작 (LLM_EARLY_MS=300)
  python eval/experiment.py --out runs/exp5 --tts stream,stream+ack,headtail_hedge2+early+ack   # +ack = 함수 호출 전에 확인 문구 먼저 (TOOL_ACK=1)
  python eval/experiment.py --out runs/exp6 --tts stream,stream+ann,stream+annnum   # +ann = Jev 작업 안내(번호 없이, 미리 만든 음성) / +annnum = 번호 넣은 안내
말하기 설정 이름은 eval/tts_presets.py 와 같습니다. 폴더 이름은 '<말하기>-<업무 분류>' (예: stream-off = 예전 서버와 같음)
결과: <out>/<설정>/results.jsonl (측정기), <out>/<설정>/server.jsonl (서버 단계별 시간), <out>/summary.md
"""
import argparse, json, os, subprocess, sys, threading, time
from collections import deque
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tts_presets import PRESETS, requests_per_answer

ROOT = Path(__file__).resolve().parents[1]
TABLE = [(1.5, 100), (2.0, 90), (2.5, 80), (3.0, 70), (3.5, 60), (4.0, 50), (5.0, 30), (6.0, 10)]


def pts(t):
    if t is None or t < 0:
        return 0
    return next((v for bound, v in TABLE if t <= bound), 0)


def read_jsonl(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    for enc in ("utf-8", "cp949"):                               # 예전 run_eval 은 Windows 에서 cp949 로 쓴다
        try:
            return [json.loads(l) for l in raw.decode(enc).splitlines() if l.strip()]
        except UnicodeDecodeError:
            continue
    return []


class Proc:
    """서브 프로세스를 띄우고 출력을 파일과 화면에 흘려 보낸다. ready 문구가 나오면 준비 완료로 본다."""

    def __init__(self, args, log, env=None, ready=None, tag=""):
        self.ready_text, self.ready, self.tag = ready, threading.Event(), tag
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log, "a", encoding="utf-8")
        self.p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace")
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.log.write(line); self.log.flush()
            print(f"{self.tag}{line}", end="", flush=True)
            if self.ready_text and line.startswith(self.ready_text):
                self.ready.set()

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(10)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.log.close()


def tts_requests(rec):
    """서버 기록 한 줄에서 실제로 보낸 Gemini 요청 수 (고정 문구는 0)."""
    t = rec.get("tts") or {}
    return 0 if t.get("cached") else len(t.get("segments") or [])


def summarize(out, modes, cases):
    """modes 는 설정 이름 목록 (폴더 이름)."""
    lines = ["# 실제 음성 실험 결과", "", "| 문항 | " + " | ".join(f"{m} 첫 소리(s) / 점수" for m in modes) + " |",
             "|---|" + "---:|" * len(modes)]
    data = {m: {r["case_id"]: r for r in read_jsonl(out / m / "results.jsonl")} for m in modes}
    logs = {m: {r["case_id"]: r for r in read_jsonl(out / m / "server.jsonl")} for m in modes}
    totals = {m: [] for m in modes}
    for c in cases:
        cells = []
        for m in modes:
            r = data[m].get(c)
            if not r or r.get("error"):
                cells.append(f"오류 {r.get('error', '없음')[:30] if r else '결과 없음'}")
                totals[m].append(0)
                continue
            fs, after = r.get("first_sound"), r.get("after_tool_sound")
            sp = 0 if r.get("spoke_before_end") else pts(fs)
            score = (sp + pts(after)) / 2 if r.get("tool_calls") else sp
            totals[m].append(score)
            extra = f", 조회 뒤 {after:.2f}s" if after is not None else ""
            laps = (logs[m].get(c) or {}).get("laps", [])
            names = [k for k, _ in laps]
            upto = names[: names.index("첫 소리 전송") + 1] if "첫 소리 전송" in names else []
            cut = fs is not None and upto and fs - sum(v for k, v in laps if k in upto) < 0
            cells.append((f"{fs:.2f}{extra} / {score:g}" if fs is not None else "소리 없음 / 0") + (" ⚠말 도중 끊음" if cut else ""))
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines.append("| 평균 점수 | " + " | ".join(f"{sum(v) / len(v):.1f}" if v else "-" for v in totals.values()) + " |")
    lines += ["", "점수는 공개 속도표(1.5초 이하 100점 … 6초 초과 0점), 함수 호출 문항은 첫 소리와 조회 뒤 답의 평균입니다. "
              "'말 도중 끊음'은 서버가 손님 말 중간의 쉼을 말 끝으로 잘못 판단한 문항이라 점수를 그대로 믿으면 안 됩니다."]
    lines += ["", "## 서버 안에서 걸린 시간 (말 끝 감지 뒤, 초)", ""]
    for m in modes:
        lines.append(f"### {m}")
        lines.append("| 문항 | 말 끝 감지(실제 끝 뒤) | 말 끝 판정(무음 ms·Jev 질문 수) | 업무 판정 | 판정 대기 | 받아쓰기 | 미리 만든 답 | 생각 | 프롬프트·입력 토큰·LLM 호출(ms) | 첫 소리 전송 | TTS 조각 | TTS 첫 음성 | 전부 준비 | 끊김 |")
        lines.append("|---|---:|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|")
        for c in cases:
            r = logs[m].get(c)
            if not r:
                continue
            laps = dict((k, v) for k, v in r.get("laps", []))
            case, t = r.get("case") or {}, r.get("tts") or {}
            sec = lambda ms: f"{ms / 1000:.2f}" if isinstance(ms, (int, float)) else "-"
            fs = (data[m].get(c) or {}).get("first_sound")
            names = [k for k, _ in r.get("laps", [])]
            upto = names[: names.index("첫 소리 전송") + 1] if "첫 소리 전송" in names else []
            detect = fs - sum(laps[k] for k in upto) if (fs is not None and upto) else None   # 실제 말 끝 → 서버가 말 끝으로 판단한 시각
            det = "-" if detect is None else (f"{detect:.2f}" + (" (말 도중 끊음)" if detect < 0 else ""))
            e = r.get("eot") or {}
            eot_cell = f"{e.get('by', '-')} ({e.get('silent_ms', '-')}·{len(e.get('asks') or [])})"
            lines.append(f"| {c} | {det} | {eot_cell} | {case.get('case')} ({case.get('top')}, {'미리' if case.get('early') else '확정 뒤'}) | "
                         f"{sec(case.get('wait_ms'))} | {laps.get('받아쓰기 확정', '-')} | "
                         f"{('사용, 대기 ' + str(laps.get('미리 만든 답 대기', 0)) + 's') if (r.get('draft') or {}).get('used') else ('버림' if (r.get('draft') or {}).get('made') else '-')} | "
                         f"{laps.get('생각', '-')} | "
                         f"{(r.get('llm') or {}).get('prompt_mode', '-')} · {[x.get('in') for x in (r.get('llm') or {}).get('llm', [])]} · "
                         f"{[round(x.get('ms', 0)) for x in (r.get('llm') or {}).get('llm', [])]} | "
                         f"{laps.get('첫 소리 전송', '-')} | {t.get('parts')}{' 고정' if t.get('cached') else ''}"
                         f"{' (재사용 ' + str(sum(1 for x in t.get('segments', []) if x.get('cached'))) + ')' if any(x.get('cached') for x in t.get('segments', [])) else ''} | "
                         f"{sec(t.get('first_audio_ms'))} | {sec(t.get('all_ready_ms'))} | {sec(t.get('stall_ms'))} |")
        lines.append("")
    lines += ["## 대화 내용", ""]
    for c in cases:
        for m in modes:
            ack = (logs[m].get(c) or {}).get("tool_ack")
            if ack:
                lines.append(f"- {c} [{m}] 함수 호출 전 확인 문구: {ack}")
            ann = (logs[m].get(c) or {}).get("announce")
            if ann:
                state = "보냄" if ann.get("played") else ("버림: " + ann["skipped"] if ann.get("skipped") else "안 보냄(본답이 먼저 준비됨)")
                lines.append(f"- {c} [{m}] 안내 문장({ann.get('action')}): {ann.get('sentence')} → {state}")
    for c in cases:
        for m in modes:
            r = logs[m].get(c) or {}
            if r.get("customer") is not None:
                lines.append(f"- {c} [{m}] 고객: {r.get('customer')} → 상담원: {r.get('reply')}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--cases", default="data/ko_cases4.jsonl")
    ap.add_argument("--audio", default="data/audio/ko")
    ap.add_argument("--tts", default="stream,split_k2,headtail_hedge2", help="말하기 설정 (eval/tts_presets.py 이름, 뒤에 +short +ann +eot 등 가능)")
    ap.add_argument("--routers", default="jev", help="업무 분류 CASE_ROUTER (off = 예전 서버, 쉼표로 여러 개)")
    ap.add_argument("--exclude", default="KO11", help="뺄 문항 id (쉼표로 구분)")
    ap.add_argument("--port", type=int, default=9300, help="이 PC 는 9000번이 막혀 있어 9300 을 기본으로 씀")
    ap.add_argument("--mock-port", type=int, default=8100)
    ap.add_argument("--rpm", type=int, default=0, help="분당 최대 Gemini TTS 요청 수 (0 = 제한 없음)")
    ap.add_argument("--call-timeout", type=int, default=90)
    a = ap.parse_args()
    out = ROOT / a.out
    skip = set(a.exclude.split(",")) if a.exclude else set()
    cases = [json.loads(l)["case_id"] for l in open(ROOT / a.cases, encoding="utf-8") if l.strip()]
    cases = [c for c in cases if c not in skip]
    configs = []
    for t in (t for t in a.tts.split(",") if t):
        base, flags = t.split("+")[0], set(t.split("+")[1:])
        env = dict(PRESETS[base][1]) if base in PRESETS else {"TTS_MODE": base}
        env["SHORT_FIRST"] = "1" if "short" in flags else "0"
        env["LLM_EARLY_MS"] = "300" if "early" in flags else "0"
        env["TOOL_ACK"] = "1" if "ack" in flags else "0"
        env["ANNOUNCE"] = "number" if "annnum" in flags else ("static" if "ann" in flags else "off")
        env["EOT_MODE"] = "jev" if "eot" in flags else "off"          # +eot: Jev 로 말 끝 판정 (없으면 예전처럼 무음만)
        for r in (r for r in a.routers.split(",") if r):
            configs.append((f"{t.replace('+', '_')}-{r}", {**env, "CASE_ROUTER": r}, base))
    modes = [name for name, _, _ in configs]
    per_call = {name: (requests_per_answer(base, 2) if base in PRESETS else 3) for name, _, base in configs}
    print(f"계획: {len(cases)}문항 {cases} × 설정 {modes} = 통화 {len(cases) * len(configs)}번", flush=True)
    mock_url = f"http://127.0.0.1:{a.mock_port}"
    mock = None
    try:
        httpx.get(mock_url + "/tools", timeout=2).raise_for_status()
        print(f"전산실이 이미 켜져 있어 그대로 씁니다: {mock_url}", flush=True)
    except Exception:
        mock = Proc([sys.executable, "mock_tools/mock_api_template.py", "--port", str(a.mock_port)], out / "mock.log", tag="[전산실] ")
        time.sleep(1.5)
    window = deque()
    try:
        for m, preset_env, _ in configs:
            env = {**os.environ, **preset_env, "SERVER_LOG": str(out / m / "server.jsonl"),
                   "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "VERBOSE": os.getenv("VERBOSE", "1")}
            srv = Proc([sys.executable, "agent_server/server.py", "--port", str(a.port)], out / m / "server.log", env=env,
                       ready="[준비 완료]", tag=f"[{m}] ")
            try:
                t0 = time.time()
                while not srv.ready.wait(0.5):
                    if srv.p.poll() is not None or time.time() - t0 > 120:
                        raise SystemExit("서버가 준비되지 않았습니다 (꺼졌거나 120초 초과). 위 출력이나 server.log 를 확인하세요.")
                for c in cases:
                    if c in {r["case_id"] for r in read_jsonl(out / m / "results.jsonl")}:
                        continue                                  # 이미 한 문항은 건너뛴다 (중간에 멈췄다 다시 돌릴 때)
                    while a.rpm > 0:                              # --rpm 을 줬을 때만: 최근 60초 요청 + 이번 통화 최대치 <= rpm
                        now = time.time()
                        while window and now - window[0] > 60:
                            window.popleft()
                        if len(window) + per_call[m] <= a.rpm:
                            break
                        wait = 60 - (now - window[0]) + 0.5
                        print(f"   (--rpm {a.rpm} 에 맞춰 {wait:.0f}초 쉽니다)", flush=True)
                        time.sleep(wait)
                    before = len(read_jsonl(out / m / "server.jsonl"))
                    one = out / m / "_one.jsonl"
                    one.write_text(next(l for l in open(ROOT / a.cases, encoding="utf-8") if json.loads(l)["case_id"] == c), encoding="utf-8")
                    try:
                        subprocess.run([sys.executable, "eval/run_eval.py", "--server", f"ws://127.0.0.1:{a.port}", "--cases", str(one),
                                        "--audio", a.audio, "--mock", mock_url, "--out", str(out / m), "--concurrency", "1"],
                                       cwd=ROOT, timeout=a.call_timeout, env={**os.environ, "PYTHONUTF8": "1"})
                    except subprocess.TimeoutExpired:
                        print(f"   {c}: {a.call_timeout}초 안에 끝나지 않아 다음 문항으로 넘어갑니다", flush=True)
                    time.sleep(0.5)
                    recs = read_jsonl(out / m / "server.jsonl")[before:]
                    window.extend([time.time()] * sum(tts_requests(r) for r in recs))
            finally:
                srv.stop()
                (out / m / "_one.jsonl").unlink(missing_ok=True)
    finally:
        if mock:
            mock.stop()
    print("\n" + summarize(out, modes, cases), flush=True)
    print(f"\n결과: {out / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
