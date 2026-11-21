# [업무별 프롬프트 효과] 같은 고객 말로 LLM 을 프롬프트 방식별로 불러, 답이 나오기까지 걸린 시간과 입력 토큰을 비교합니다.
# 왜 있나: Jev 가 고른 업무에 맞춰 프롬프트를 바꾸면 LLM 응답이 실제로 빨라지는지(답 품질은 그대로인지) 확인하려고요.
# 업무 판정은 정답 라벨을 그대로 써서, 분류가 틀린 영향 없이 프롬프트 차이만 봅니다. 분류 정확도는 bench_case.py 로 따로 봅니다.
"""사용 (저장소 맨 위에서):
  python eval/bench_llm.py --out runs/llm1                   # 문장 20개 × full, add, slim × 2번
  python eval/bench_llm.py --out runs/llm2 --variants full,slim --reps 3
방식
  full  업무를 정하지 않음: 매뉴얼 전체 + 도구 전부 (지금 서버와 같음)
  add   매뉴얼 전체 + 업무 절차 + 도구 전부
  slim  공통 규칙 + 업무 절차 + 그 업무 도구만
함수 호출이 있는 문장은 전산실(mock_tools, 8100)을 거칩니다. 꺼져 있으면 이 스크립트가 켰다가 끕니다.
결과: <out>/llm_results.jsonl, <out>/llm_summary.md
"""
import argparse, asyncio, json, statistics, subprocess, sys, time, uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent_server"))
import server
import case_router


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs)) if xs else None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/llm_bench")
    ap.add_argument("--items", default="eval/case_items.jsonl")
    ap.add_argument("--exclude", default="KO11", help="뺄 문항 id (쉼표로 구분)")
    ap.add_argument("--variants", default="full,add,slim")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--mock-port", type=int, default=8100)
    a = ap.parse_args()
    skip = set(a.exclude.split(",")) if a.exclude else set()
    items = [json.loads(l) for l in open(ROOT / a.items, encoding="utf-8") if l.strip()]
    items = [it for it in items if it["id"] not in skip]
    variants = [v for v in a.variants.split(",") if v]
    print(f"계획: 문장 {len(items)}개 × {variants} × {a.reps}번 = LLM 답 {len(items) * len(variants) * a.reps}개 "
          f"(함수 호출 문장은 LLM 을 2번 부름)", flush=True)
    mock_url, mock = f"http://127.0.0.1:{a.mock_port}", None
    try:
        httpx.get(mock_url + "/tools", timeout=2).raise_for_status()
    except Exception:
        mock = subprocess.Popen([sys.executable, "mock_tools/mock_api_template.py", "--port", str(a.mock_port)], cwd=ROOT,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        base = {"now": "2026-11-21 11:00", "history": [], "caller": None, "tools_url": mock_url}
        tools = await server.load_tools({**base, "case_id": "bench"})
        await server.ai.chat.completions.create(model=server.LLM_MODEL, messages=[{"role": "user", "content": "안녕하세요"}])   # 연결 예열 (기록 안 함)
        with open(out / "llm_results.jsonl", "a", encoding="utf-8") as f:
            for rep in range(1, a.reps + 1):
                for i, it in enumerate(items):
                    order = variants[(i + rep) % len(variants):] + variants[: (i + rep) % len(variants)]   # 순서를 돌려 시간대 영향을 고르게
                    for v in order:
                        case = None
                        if v != "full":
                            cid = it["expected"] if it["expected"] in case_router.GUIDES else None
                            case = case_router.Decision(case_id=cid, method="label", top=it["expected"]) if cid else None
                        start = {**base, "case_id": it["id"], "session_id": f"bench-{uuid.uuid4().hex[:8]}"}
                        stats, t, error = {}, time.perf_counter(), None
                        try:
                            reply = await server.think(start, it["text"], tools, case, stats, mode=None if v == "full" else v)
                        except Exception as e:
                            reply, error = "", repr(e)[:200]
                        total = round((time.perf_counter() - t) * 1000, 1)
                        llm = stats.get("llm", [])
                        row = {"id": it["id"], "variant": v, "rep": rep, "case": it["expected"], "total_ms": total,
                               "rounds": len(llm), "first_llm_ms": llm[0]["ms"] if llm else None,
                               "in_tokens": llm[0]["in"] if llm else None, "cached_tokens": llm[0]["cached"] if llm else None,
                               "system_chars": stats.get("system_chars"), "tools": stats.get("tools"), "calls": stats.get("calls"),
                               "reply": reply, "error": error}
                        rows.append(row)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                        print(f"{it['id']} {v:<4} {total:>7.0f}ms LLM {len(llm)}번 입력 {row['in_tokens']}토큰 "
                              f"함수 {[c['name'] for c in row['calls'] or []]} | {reply[:50]}{' 오류 ' + error if error else ''}", flush=True)
    finally:
        if mock:
            mock.terminate()
    lines = ["| 방식 | 답 수 | 전체 시간 중앙값(ms) | 첫 LLM 호출 중앙값(ms) | 함수 없는 문장 중앙값(ms) | 함수 있는 문장 중앙값(ms) | 입력 토큰 중앙값 | 평균 LLM 호출 수 | 오류 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in variants:
        rs = [r for r in rows if r["variant"] == v and not r["error"]]
        lines.append(f"| {v} | {len(rs)} | {med([r['total_ms'] for r in rs])} | {med([r['first_llm_ms'] for r in rs])} | "
                     f"{med([r['total_ms'] for r in rs if r['rounds'] == 1])} | {med([r['total_ms'] for r in rs if r['rounds'] > 1])} | "
                     f"{med([r['in_tokens'] for r in rs])} | {statistics.mean([r['rounds'] for r in rs]) if rs else 0:.2f} | "
                     f"{sum(1 for r in rows if r['variant'] == v and r['error'])} |")
    lines += ["", "## 문장별 답 (1회차)", ""]
    for it in items:
        for v in variants:
            r = next((r for r in rows if r["id"] == it["id"] and r["variant"] == v and r["rep"] == 1), None)
            if r:
                calls = ", ".join(f"{c['name']}({c['arguments']})" for c in r["calls"] or [])
                lines.append(f"- {it['id']} [{v}] {r['total_ms']:.0f}ms{' 함수 ' + calls if calls else ''}: {r['reply']}")
    text = "\n".join(lines) + "\n"
    (out / "llm_summary.md").write_text(text, encoding="utf-8")
    print("\n" + "\n".join(lines[: len(variants) + 2]) + f"\n\n{server.usage_line()}\n결과: {out / 'llm_summary.md'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
