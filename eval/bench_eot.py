# [말 끝 판정 비교] 고객이 쉬는 순간의 받아쓴 글을 Jev 에 보내 end/hold 를 맞히는지, 얼마나 빨리 답하는지 잽니다.
# 왜 있나: 서버는 쉬는 동안 일정 간격으로 Jev 에 묻고, end 면 EOT_END_MS 에 끊습니다. 말 중간 쉼에서 end 가 나오면 고객 말을 끊게 되니 미리 확인합니다.
# 비용: 문장 수 × 무음 길이 수 × 반복 수 만큼 Jev 요청 (기본 29 × 2 × 1 = 58회).
"""사용 (저장소 맨 위에서):
  python eval/bench_eot.py --out runs/eot1
  python eval/bench_eot.py --out runs/eot2 --silences 300,600,1000 --repeat 2
결과: <out>/eot_results.jsonl, <out>/eot_summary.md
'서버 판정'은 Jev 답에 서버의 안전장치(인사만 했거나 '~인데요'처럼 끝나면 end 를 믿지 않음)를 더한 최종 결과입니다.
"""
import argparse, asyncio, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent_server"))
import server                                                 # .env 읽기, httpx 클라이언트를 같이 쓴다
import turn_judge


GATE = None


async def ask(item, silence_ms):
    async with GATE:                                          # 한꺼번에 너무 많이 보내지 않는다 (동시 6개)
        return await _ask(item, silence_ms)


async def _ask(item, silence_ms):
    j = turn_judge.TurnJudge(server.http, item.get("history"))
    j.on = True
    t = time.perf_counter()
    await j._ask(1, 0, "x", item["text"], silence_ms)
    seq, epoch, key, s, choice, prob, ms = j.answers[0]
    err = j.log[0]["error"]
    sure = choice if (prob or 0) >= turn_judge.EOT_MIN_PROB else None
    final = None if (sure == "end" and turn_judge.end_vetoed(item["text"])) else sure
    return {"id": item["id"], "silence_ms": silence_ms, "expected": item["expected"], "jev": choice, "prob": prob,
            "server": final or "fallback", "ms": ms, "error": err, "text": item["text"]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/eot_bench")
    ap.add_argument("--items", default="eval/eot_items.jsonl")
    ap.add_argument("--silences", default="300,600")
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    global GATE
    GATE = asyncio.Semaphore(6)
    if not turn_judge.os.getenv("TYPESAFE_API_KEY"):
        raise SystemExit("TYPESAFE_API_KEY 가 없습니다 (.env 확인)")
    items = [json.loads(l) for l in open(ROOT / a.items, encoding="utf-8") if l.strip()]
    silences = [int(s) for s in a.silences.split(",")]
    rows = []
    for _ in range(a.repeat):
        for s in silences:
            rows += await asyncio.gather(*(ask(it, s) for it in items))
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "eot_results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ms = sorted(r["ms"] for r in rows if not r["error"])
    q = lambda p: ms[min(len(ms) - 1, int(round(p * (len(ms) - 1))))] if ms else None
    lines = ["# 말 끝 판정 (Jev) 결과", "",
             f"요청 {len(rows)}회, 오류 {sum(1 for r in rows if r['error'])}회, 걸린 시간 중앙값 {q(0.5)}ms · p95 {q(0.95)}ms · 최대 {ms[-1] if ms else '-'}ms", "",
             "| 무음(ms) | Jev 정답 | 말 중간에 end (끊을 위험) | 그중 서버 안전장치가 막음 | 끝났는데 hold (늦어짐) | 확신 낮음 |",
             "|---:|---:|---:|---:|---:|---:|"]
    for s in silences:
        rs = [r for r in rows if r["silence_ms"] == s and not r["error"]]
        ok = sum(1 for r in rs if r["jev"] == r["expected"])
        cut = [r for r in rs if r["expected"] == "hold" and r["jev"] == "end" and (r["prob"] or 0) >= turn_judge.EOT_MIN_PROB]
        blocked = sum(1 for r in cut if r["server"] == "fallback")
        late = sum(1 for r in rs if r["expected"] == "end" and r["server"] == "hold")
        low = sum(1 for r in rs if (r["prob"] or 0) < turn_judge.EOT_MIN_PROB)
        lines.append(f"| {s} | {ok}/{len(rs)} | {len(cut)} | {blocked} | {late} | {low} |")
    lines += ["", "## 문장별", "", "| id | 무음 | 정답 | Jev (확률) | 서버 판정 | ms | 글 |", "|---|---:|---|---|---|---:|---|"]
    for r in rows:
        mark = "" if r["jev"] == r["expected"] else " ✗"
        p = f"{r['prob']:.2f}" if r["prob"] is not None else "-"
        lines.append(f"| {r['id']} | {r['silence_ms']} | {r['expected']} | {r['jev']} ({p}){mark} | {r['server']} | {r['ms']} | {r['text']} |")
    lines += ["", f"서버는 end 를 받으면 무음 {turn_judge.EOT_END_MS}ms 에, hold 면 {turn_judge.EOT_HOLD_MS}ms 에, "
              f"판정이 없거나(fallback) 확신이 낮으면 무음 {server.END_SILENCE_MS}ms 에 끊습니다. "
              "말 중간 쉼은 대부분 0.6초 아래라서, '말 중간에 end' 가 나와도 그 쉼이 EOT_END_MS 보다 짧으면 실제로 끊기지는 않습니다."]
    (out / "eot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    asyncio.run(main())
