# [meta.json 자동 생성기] wav 폴더를 넣으면 각 녹음에서 "고객이 말을 끝낸 시각"을 계산해 meta.json을 만듭니다.
# 왜 있나: run_eval 의 스톱워치가 정확하려면 이 값이 맞아야 하는데, 직접 재기 번거로워서 만든 도구예요.
"""wav 파일에서 '고객이 말을 끝낸 시각'을 계산해 meta.json 을 만든다. (run_eval.py 가 읽는 파일)

사용:  python eval/make_meta.py data/audio/demo
       python eval/make_meta.py my_audio --threshold 300

규칙: 16kHz / 16-bit / mono wav 이고, 파일 이름(DEMO1.wav → "DEMO1")이 case_id 와 같아야 한다.
이미 meta.json 이 있으면 wav 가 있는 항목만 새 값으로 바꾸고, 나머지는 그대로 둔다.
"""
import argparse, json, wave
from pathlib import Path

import numpy as np

RATE = 16000


def speech_end(path, threshold):
    """마지막으로 소리가 있는 20ms 조각의 끝 시각(초). 소리가 없으면 None."""
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
            raise ValueError("16kHz / mono / 16-bit wav 가 아닙니다")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    n = RATE // 50                                              # 20ms = 320 샘플
    frames = pcm[: len(pcm) // n * n].reshape(-1, n)
    loud = np.sqrt((frames ** 2).mean(axis=1)) >= threshold     # 조각마다 볼륨(RMS)이 기준 이상인가
    return round((np.flatnonzero(loud)[-1] + 1) * 0.02, 3) if loud.any() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio_dir", help="wav 파일이 들어 있는 폴더 (여기에 meta.json 이 만들어집니다)")
    ap.add_argument("--threshold", type=float, default=300, help="이 볼륨(RMS) 이상이면 말소리 (기본 300, run_eval 과 같음)")
    args = ap.parse_args()

    folder = Path(args.audio_dir)
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"wav 파일이 없습니다: {folder}")
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    for wav in wavs:
        try:
            end = speech_end(wav, args.threshold)
        except (ValueError, wave.Error, EOFError) as e:           # EOFError: 비어 있는 파일
            print(f"건너뜀  {wav.name}: {str(e) or '읽을 수 없는 wav 파일입니다'}")
            continue
        if end is None:
            print(f"건너뜀  {wav.name}: 소리가 없습니다 (--threshold 를 낮춰 보세요)")
            continue
        old = meta.get(wav.stem, {}).get("speech_end")
        meta.setdefault(wav.stem, {})["speech_end"] = end
        print(f"{wav.stem}: speech_end = {end}s" + (f"   (이전 값 {old}s)" if old not in (None, end) else ""))

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {meta_path}")


if __name__ == "__main__":
    main()
