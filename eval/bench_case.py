# [업무 유형 분류 비교] 고객 첫 말 20개(KO11 제외)를 방식별(키워드 / Jev / LLM)로 분류해 업무·바로 시작할 작업의 정확도와 걸린 시간을 잽니다.
# 왜 있나: Jev 로 업무를 고르는 게 키워드보다 정확한지, 말 끝 판정을 기다리는 동안(약 0.5초) 끝나는지 확인하려고요.
# 비용: 방식마다 문장 수 × 반복 수 만큼 요청합니다 (keyword 는 0회). 기본은 keyword + jev 1번씩 = Jev 20회.
"""사용 (저장소 맨 위에서):
  python eval/bench_case.py --out runs/case1                       # keyword, jev
  python eval/bench_case.py --out runs/case2 --methods keyword,jev,llm
주의: keyword 의 정확도는 낙관적입니다. cases.json 의 키워드를 KO01~KO11 문장을 보면서 적었기 때문입니다.
결과: <out>/case_results.jsonl, <out>/case_summary.md
"""
import argparse, asyncio, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent_server"))
import server                                                 # .env 읽기, httpx·OpenAI 클라이언트를 같이 쓴다
import case_router


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/case_bench")
    ap.add_argument("--items", default="eval/case_items.jsonl")
    ap.add_argument("--methods", default="keyword,jev")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--exclude", default="KO11", help="뺄 문항 id (쉼표로 구분)")
    a = ap.parse_args()
    skip = set(a.exclude.split(",")) if a.exclude else set()
    items = [json.loads(l) for l in open(ROOT / a.items, encoding="utf-8") if l.strip()]
    items = [it for it in items if it["id"] not in skip]
    methods = [m for m in a.methods.split(",") if m]
    paid = sum(len(items) * a.reps for m in methods if m != "keyword")
    print(f"계획: 문장 {len(items)}개 × 방식 {methods} × {a.reps}번 → 유료 요청 약 {paid}회", flush=True)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "case_results.jsonl", "a", encoding="utf-8") as f:
        for rep in range(1, a.reps + 1):
            for it in items:
                for m in methods:                                 # 하나씩 차례로 보내 걸린 시간을 깨끗하게 잰다
                    d = await case_router.classify(it["text"], [], method=m, http=server.http, ai=server.ai,
                                                   model=server.LLM_MODEL, on_llm_usage=server.count_llm)
                    ok = d.top == it["expected"]
                    sentence, number = case_router.announcement(d.action, it["text"], "number")
                    row = {"id": it["id"], "method": m, "rep": rep, "expected": it["expected"], "top": d.top, "prob": d.prob,
                           "confidence": d.confidence, "strict": ok, "lenient": ok or d.top in it.get("also_ok", []),
                           "latency_ms": d.latency_ms, "error": d.error, "used_method": d.method,
                           "probabilities": d.probabilities, "action": d.action, "action_prob": d.action_prob,
                           "action_expected": it.get("action"), "action_ok": d.action == it.get("action"),
                           "number": number, "number_ok": number == it.get("number"), "announce": sentence}
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                    p = f" {d.prob:.2f}" if d.prob is not None else ""
                    mark = "O" if row["strict"] else ("△" if row["lenient"] else "X")
                    amark = "O" if row["action_ok"] else "X"
                    print(f"{it['id']} {m:<7} {mark} {d.top}{p} (정답 {it['expected']}) | 작업 {amark} {d.action} (정답 {it.get('action')}) "
                          f"| 안내: {sentence or '-'} | {d.latency_ms:.0f}ms{' ' + d.error if d.error else ''}", flush=True)
    lines = ["| 방식 | 업무 정확(엄격) | 업무 정확(허용) | 작업 정확 | 안내 번호 정확 | 지연 p50(ms) | p95(ms) | 최대(ms) | 오류 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        lat = [r["latency_ms"] for r in rs if not r["error"]]
        lines.append(f"| {m} | {sum(r['strict'] for r in rs)}/{len(rs)} | {sum(r['lenient'] for r in rs)}/{len(rs)} | "
                     f"{sum(r['action_ok'] for r in rs)}/{len(rs)} | {sum(r['number_ok'] for r in rs)}/{len(rs)} | "
                     f"{pct(lat, .5)} | {pct(lat, .95)} | {pct(lat, 1)} | {sum(bool(r['error']) for r in rs)} |")
    misses = [f"- {r['method']} {r['id']}: 업무 {r['top']} (정답 {r['expected']})" for r in rows if not r["strict"]]
    misses += [f"- {r['method']} {r['id']}: 작업 {r['action']} (정답 {r['action_expected']})" for r in rows if not r["action_ok"]]
    text = "\n".join(lines) + ("\n\n틀린 것\n" + "\n".join(misses) if misses else "") + "\n"
    (out / "case_summary.md").write_text(text, encoding="utf-8")
    print("\n" + text + f"\n결과: {out / 'case_summary.md'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
