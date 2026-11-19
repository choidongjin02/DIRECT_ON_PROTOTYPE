# DIRECT:ON 2026 — 실시간 음성 AI 상담원 (1차 프로토타입)

전화로 들어온 고객 목소리를 **알아듣고 → 생각하고 → 목소리로 답하는** AI 상담원 서버입니다.
대회 지정 모델: 받아쓰기 **Soniox** · 생각 **gpt-5.4-mini** · 말하기 **Gemini 2.5 Flash TTS**

```
고객 음성 → 상담원 서버(9000번) → Soniox(받아쓰기) → gpt-5.4-mini ⇄ 전산실(8100번) → Gemini(말하기) → 음성 응답
```

## 현재 상태
- 끝까지 동작합니다: 음성 입력 → 주문 조회·콜백 같은 함수 호출 → 음성 응답. 오류가 나도 `turn.end`는 항상 보냅니다.
- 속도(우리 환경, 연습 문항): 미리 만들어 둔 고정 문구 답변은 첫 소리 약 1.7초, 함수 호출이 있는 답변은 약 2.7초.
- 아직 안 한 것: 11/21 매뉴얼·도구 정의 반영 (지금은 연습용 도구 2개: 주문 조회, 콜백).

## 폴더 구조
```
agent_server/       상담원 서버 (제출물)
  server.py           본체: 듣기 → 받아쓰기 → 생각 → 말하기
  prompt.md           매뉴얼과 말투 규칙 (11/21에 매뉴얼을 여기에)
  requirements.txt    패키지 목록
  .env.example        키 견본 (복사해서 .env 로)
  check_apis.py       API 3개가 되는지 점검 (개발용)
mock_tools/         가짜 주문 전산실 (8100번) — 상담원이 조회·접수 함수를 부르면 답해 주는 서버 (제출물)
eval/               시험 도구: run_eval(측정기) · record(마이크 녹음) · make_meta · score_demo(속도 점수표)
data/               연습 데이터: 도구 정의, 가짜 장부, 영어 데모 음성
docs/               주최 측 통신 규격 문서 (참고)
example_server/     삐 소리만 내는 예제 서버 (참고용)
```

## 실행 방법

### 1. 설치 (한 번만)
```bash
python -m pip install -r agent_server/requirements.txt
python -m pip install sounddevice
```

### 2. 키 넣기
`agent_server/.env.example`을 `agent_server/.env`로 복사한 뒤, 메모장으로 열어 키 3개를 적습니다. (따옴표 없이 `OPENAI_API_KEY=키값` 형태)
```powershell
copy agent_server\.env.example agent_server\.env
```
> `.env`는 **절대 깃에 올리지 마세요.** (`.gitignore`로 막아 두었습니다)

### 3. 키가 되는지 점검 (크레딧 1센트 안팎)
```bash
python agent_server/check_apis.py
```
STT → LLM → TTS 순서로 한 번씩 불러 봅니다. 모두 "성공"이면 준비 끝입니다.

### 4. 내 목소리 녹음 (약 10분)
```bash
python eval/record.py
```
문장이 나오면 **Enter → 읽기 → Enter → Enter(저장)** 순서입니다. 11문장을 녹음하면 `data/audio/ko/`와 `data/ko_cases.jsonl`이 만들어집니다. (**KO11은 중간에 1초 넘게 쉬세요**)

### 5. 서버 켜기 (터미널 2개)
```bash
python mock_tools/mock_api_template.py --port 8100
```
```powershell
$env:PYTHONUTF8="1"; python agent_server/server.py --port 9000
```
`[준비 완료]`가 나올 때까지 기다리세요. (macOS/Linux는 `PYTHONUTF8=1 python agent_server/server.py --port 9000`)

### 6. 전화 걸어 보기 (터미널 1개 더)
처음엔 끝에 `--limit 1`을 붙여 한 문장만 시험해 보세요.
```powershell
$env:PYTHONUTF8="1"; python eval/run_eval.py --server ws://127.0.0.1:9000 --cases data/ko_cases.jsonl --audio data/audio/ko --mock http://127.0.0.1:8100 --out runs/ko1 --concurrency 1
```
```bash
python eval/score_demo.py --run runs/ko1
```
- `--out` 이름은 **매번 새로** 정하세요. (같은 이름이면 이미 한 문항은 건너뜁니다)
- `--concurrency 1`은 **한 번에 한 통씩** 거는 옵션입니다. Gemini TTS가 분당 10회 한도(Tier 1)라서 필요해요. `429` 오류가 나면 1분 쉬었다가 다시 하세요.
- 결과는 **상담원 터미널**에서 봅니다: `고객:`(받아쓴 글) · `상담원:`(대답) · 단계별 시간 · `[사용량]`(추정 비용). 상담원 목소리는 `runs/<이름>/agent_audio/`에 저장됩니다.

## 설정값 (`agent_server/.env`에 적거나 터미널에서 지정)
| 이름 | 기본값 | 설명 |
|---|---|---|
| `LLM_EFFORT` | (비움) | LLM 생각 강도 `low`/`medium` (높을수록 정확하지만 느림) |
| `END_SILENCE_MS` | 800 | 이만큼 조용하면 말이 끝났다고 판단 |
| `LONG_SILENCE_MS` | 1800 | 말이 `~는데`, `~하고`처럼 이어질 모양일 때 최대로 더 기다리는 시간 |
| `TTS_STREAM` | 1 | 목소리를 조각으로 받는 대로 바로 내보내기 (0이면 끔) |
| `TTS_PER_SENTENCE` | 0 | 1이면 문장마다 따로 목소리 요청 (요청 수가 늘어 한도에 불리) |
| `TTS_VOICE` / `TTS_STYLE` | Kore / `Say: ` | 목소리 종류 / 읽어 달라는 지시 (지시가 없으면 Gemini가 실패함) |
| `VERBOSE` | 1 | 0이면 고객 발화를 출력하지 않음 (**정식 측정 때는 0**) |

## 알아 둘 것
- **11/21에 할 일**: 매뉴얼을 `prompt.md`에 붙여 넣기 · 자주 쓰는 고정 문구(`server.py`의 `PHRASES`)를 매뉴얼 문구로 바꾸기 · 도구 정의(`data/demo_tools.json`)와 가짜 장부(`data/demo_mock_data.json`) 교체 · `history` 형식 확인(`server.py`의 `history_messages`).
- **알려진 한계**: Gemini 한도(분당 10회) · 스타터의 `data/audio/demo/meta.json` 값이 녹음과 안 맞음(그래서 직접 녹음한 파일을 쓴다) · 받아쓰기 오인식("교환"→"소환") 보정은 아직 없음.
- **전산실(`mock_tools/`)도 제출물입니다.** 대회 측정 때 상담원 서버와 함께 켜 두어야 하고, 측정 직전에 받는 시험용 장부 파일로 다시 띄웁니다. 겉보기엔 단순하지만 11/21에 가장 많이 바뀌는 부분입니다.
- **제출할 때 뺄 것**: `.env`(키!), `runs/`, `eval/`, `check_apis.py`, `example_server/`.
