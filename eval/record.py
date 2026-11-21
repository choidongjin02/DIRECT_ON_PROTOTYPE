# [마이크 녹음기] Enter 로 녹음을 시작·종료하면 16kHz wav 와 meta.json, 문제 파일(jsonl)을 한 번에 만들어 줍니다.
# 왜 있나: 직접 녹음한 목소리로 시험하려면 16kHz 모노 wav 와 "말 끝난 시각" 메모가 필요한데, 이걸 자동으로 해 주려고요.
"""사용:  python eval/record.py                  # 준비된 문장 11개를 차례로 보여 주며 녹음 → data/audio/ko, data/ko_cases.jsonl
        python eval/record.py KO03 KO07          # 지정한 문장만 (다시) 녹음
        python eval/record.py --file 문장.txt    # 내 문장으로 녹음 (줄마다 "ID|문장", 문장 속 ~1.2~ 는 1.2초 쉬라는 표시)
        python eval/record.py --list-devices     # 마이크 목록 (고르려면 --device 번호)
준비:  python -m pip install sounddevice
녹음 뒤:  Enter=저장   r=다시 녹음   p=들어 보기   s=건너뛰기
이어서:  python eval/run_eval.py --server ws://127.0.0.1:9000 --cases data/ko_cases.jsonl --audio data/audio/ko --mock http://127.0.0.1:8100 --out runs/ko1
"""
import argparse, json, math, re, wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from make_meta import speech_end                     # 'python eval/record.py' 로 실행하면 eval 폴더가 import 경로에 들어간다

ROOT = Path(__file__).resolve().parents[1]
RATE = 16000
DEFAULT = """# problem.html 의 4개 케이스와 그 변형
KO01|제가 주문한 상품 언제 도착해요? 주문번호는 일이삼사예요.
KO02|어제 취소 요청했는데 아직 환불이 안 됐어요. 언제 되는 거예요?
KO03|아까 그거 있잖아요. 그거 취소된 거 맞아요?
KO04|제품을 먹고 몸이 불편한데 어떻게 해야 하나요?
# 숫자 표현, 정정, 번호 없음
KO05|주문번호는 천이백삼십사입니다. 배송 상태 알려 주세요.
KO06|콜백 부탁드려요. 제 번호는 공일공 공공공공 일이삼사예요.
KO07|주문번호는 1234... 아니, 1243이요. 배송 어디쯤인가요?
KO08|제 주문이 어디쯤 왔는지 알려 주세요.
# 다른 업무 유형
KO09|교환하고 싶은데 어떻게 하면 되나요?
KO10|이 제품은 하루에 몇 번 먹어야 하나요?
# 말 중간에 1.2초 쉬는 고객 (눈치 시험)
KO11|주문 취소하고 싶은데요 ~1.2~ 결제는 카드로 했어요.
"""


def capture(device):
    """Enter 로 시작해 Enter 로 끝낼 때까지 마이크 소리를 모아 (int16 소리, 샘플레이트)를 돌려준다."""
    import sounddevice as sd                         # 녹음할 때만 필요하다
    rate = int(sd.query_devices(device, "input")["default_samplerate"])
    chunks = []
    with sd.InputStream(samplerate=rate, channels=1, dtype="int16", device=device,
                        callback=lambda data, frames, t, status: chunks.append(data[:, 0].copy())):
        input("   ● 녹음 중... 다 말했으면 Enter")
    return (np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)), rate


def to_16k(audio, rate):
    if rate != RATE:
        g = math.gcd(RATE, rate)
        audio = resample_poly(audio.astype(np.float32), RATE // g, rate // g)
    return np.clip(audio, -32768, 32767).astype(np.int16)


def write_wav(path, pcm):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def feedback(pcm, end):
    """녹음 상태를 알려 준다. 쓸 수 없는 녹음이면 False."""
    if end is None:
        print("   ! 소리가 거의 없어요. 마이크 선택이나 음량을 확인하고 다시 녹음하세요.")
        return False
    peak = int(np.abs(pcm).max())
    print(f"   길이 {len(pcm) / RATE:.1f}초, 말이 끝난 시각 {end}초")
    if peak < 1500:
        print("   ! 소리가 작아요. 마이크에 조금 더 가까이 말해 보세요.")
    elif peak >= 32000:
        print("   ! 소리가 너무 커서 깨졌을 수 있어요. 조금 작게 말해 보세요.")
    return True


def update_files(case_id, spoken, end, out, cases_path):
    """meta.json 과 문제 파일에 이 문장을 추가(또는 갱신)한다. (한글이 깨지지 않게 ASCII 로 저장)"""
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta[case_id] = {"speech_end": end}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cases = {}
    if cases_path.exists():
        for line in cases_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases[json.loads(line)["case_id"]] = json.loads(line)
    cases[case_id] = {"case_id": case_id, "room_id": case_id, "now": "2026-11-21 11:00", "history": [],
                      "customer_utterance": spoken, "customer_profile": None}
    cases_path.write_text("\n".join(json.dumps(c) for c in cases.values()) + "\n", encoding="utf-8")


def record_one(case_id, text, args, out):
    """문장 하나를 녹음한다. 저장했으면 True."""
    spoken = re.sub(r"~[\d.]+~", " ", text).strip()
    shown = re.sub(r"~([\d.]+)~", r" (...\1초 쉬기...) ", text).strip()
    print(f"\n[{case_id}] 이렇게 말해 보세요:\n   \"{shown}\"")
    tmp = out / "_tmp.wav"
    while True:
        input("   준비되면 Enter (녹음 시작)")
        pcm = to_16k(*capture(args.device))
        write_wav(tmp, pcm)
        end = speech_end(tmp, 300)
        usable = feedback(pcm, end)
        while True:
            choice = input("   [Enter]=저장  r=다시 녹음  p=들어 보기  s=건너뛰기 > ").strip().lower()
            if choice == "p":
                import sounddevice as sd
                sd.play(pcm, RATE)
                sd.wait()
                continue
            break
        if choice == "s":
            return False
        if choice == "r" or not usable:
            continue
        tmp.replace(out / f"{case_id}.wav")
        update_files(case_id, spoken, end, out, Path(args.cases))
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="녹음할 문장의 ID (없으면 전부)")
    ap.add_argument("--file", help='"ID|문장" 형식의 문장 목록 파일 (없으면 준비된 11개)')
    ap.add_argument("--device", type=int, help="마이크 번호 (--list-devices 로 확인)")
    ap.add_argument("--list-devices", action="store_true", help="마이크 목록을 보여 주고 끝낸다")
    ap.add_argument("--out", default=str(ROOT / "data" / "audio" / "ko"), help="wav 와 meta.json 을 저장할 폴더")
    ap.add_argument("--cases", default=str(ROOT / "data" / "ko_cases.jsonl"), help="문제 파일 경로")
    args = ap.parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    source = Path(args.file).read_text(encoding="utf-8") if args.file else DEFAULT
    items = [tuple(line.strip().split("|", 1)) for line in source.splitlines()
             if "|" in line and not line.lstrip().startswith("#")]
    items = [(i, t) for i, t in items if not args.ids or i in args.ids]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    try:
        for case_id, text in items:
            saved += record_one(case_id, text, args, out)
    except KeyboardInterrupt:
        print("\n중단했어요. 저장된 녹음까지는 그대로 남아 있어요.")
    finally:
        (out / "_tmp.wav").unlink(missing_ok=True)
    print(f"\n녹음 {saved}개 저장 → {out}\n문제 파일 → {args.cases}")
    print(f"이어서: python eval/run_eval.py --server ws://127.0.0.1:9000 --cases {args.cases} --audio {out} "
          f"--mock http://127.0.0.1:8100 --out runs/ko1")


if __name__ == "__main__":
    main()
