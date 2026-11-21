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
  case_router.py      (실험) 고객 첫 말로 업무 유형 분류: Jev / LLM / 키워드
  cases/              (실험) 업무 유형 목록(cases.json)과 업무별 절차(*.md) = 프롬프트 2단계
mock_tools/         가짜 주문 전산실 (8100번) — 상담원이 조회·접수 함수를 부르면 답해 주는 서버 (제출물)
eval/               시험 도구: run_eval(측정기) · record(마이크 녹음) · make_meta · score_demo(속도 점수표)
                    (실험) experiment(실제 음성 비교) · bench_tts(말하기 방식) · bench_case(업무 분류) · fake_apis(크레딧 없이 시험)
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
| `TTS_MODE` | (비움) | (실험) 말하기 방식. `stream` · `single` · `sentence_seq` · `parallel` · `parallel_ordered` · `parallel_stream`. 비우면 위 두 설정대로 |
| `TTS_MAX_PARTS` / `TTS_MIN_CHARS` / `TTS_GAP_MS` | 3 / 8 / 250 | (실험) 문장별 방식에서 최대 조각 수 / 이보다 짧은 문장은 옆에 붙임 / 이어 붙일 때 문장 사이 쉼(ms) |
| `CASE_ROUTER` | 키가 있으면 `jev` | (실험) 업무 분류 방식 `jev` · `llm` · `keyword` · `off`. `TYPESAFE_API_KEY`가 없으면 `keyword` |
| `CASE_PROMPT` | `slim` | (실험) 업무가 정해졌을 때 `slim` = 공통 규칙 + 업무 절차 + 그 업무 도구만 / `add` = 매뉴얼 전체 + 업무 절차 + 도구 전부 |
| `CASE_EARLY_MS` / `CASE_WAIT_S` / `CASE_MIN_PROB` | 300 / 0.4 / 0.5 | (실험) 말을 멈추고 이만큼 지나면 미리 분류 / 받아쓰기 확정 뒤 분류를 더 기다리는 최대 시간 / 이보다 확신이 낮으면 업무 지침을 붙이지 않음 |
| `TTS_SPLIT_CLAUSE` / `TTS_CLAUSE_GAP_MS` | 0 / 120 | (실험) 문장 수가 `TTS_MAX_PARTS` 보다 적으면 문장 안에서도 끊기: 1 = 쉼표·연결 어미(~고, ~며, ~는데 …) 뒤, 2 = 1 + 주제 조사(은·는·로는 …) 뒤. 문장 안에서 끊은 곳의 쉼(ms) |
| `TTS_KEYS` / `TTS_HEDGE` | 0 / 1 | (실험) TTS 에 쓸 Gemini 키 수(0 = `GEMINI_API_KEY`, `GEMINI_API_KEY2`, … 찾은 것 전부, 조각·통화마다 돌려 씀) / 2 이상이면 같은 조각을 그 수만큼 동시에 요청해 먼저 온 음성 사용 |
| `TTS_CACHE_ALL` | 1 | 한 번 만든 문장 음성을 서버 메모리에 두고 같은 문장이 또 나오면 바로 씀 |
| `LLM_EARLY_MS` | 0 | (실험) 0 이 아니면 말을 멈추고 이만큼(ms) 지나면 그때까지 받아쓴 글로 LLM·TTS 를 미리 시작. 확정 글이 같을 때만 그 답을 보내고, 함수는 말 끝 확정 뒤에만 부름 |
| `TOOL_ACK` | 0 | (실험) 1 이면 LLM 이 함수를 부르기로 정한 순간 도구별 확인 문구(`server.py` 의 `TOOL_ACK_PHRASES`, 미리 만든 음성)를 먼저 보내고 결과를 반영한 답을 뒤이어 보냄. 함수 호출 문항의 첫 소리를 당김 |
| `ANNOUNCE` / `ANNOUNCE_WAIT_S` | static / 3 | Jev 가 고른 '바로 시작할 작업'으로 안내 문장을 첫 소리로 보냄. `static`(기본) = 번호 없는 문장(미리 만든 음성, "주문 내역을 확인해 볼게요."), `off` = 끔, `number` = 번호 넣은 문장(실험용)("주문번호 1 5 5 7 주문 내역을 확인해 볼게요.", 말을 멈춘 직후부터 만듦). 함수를 부르기 직전 안내 음성이 아직 없으면 `ANNOUNCE_WAIT_S` 초까지 기다렸다가 안내 → 함수 순으로 보냄 |
| `SHORT_FIRST` | 0 | (실험) 1 이면 LLM 에 '두 문장 이상이면 첫 문장은 15자 안팎으로 짧게' 규칙을 더함 |
| `SERVER_LOG` | (비움) | (실험) 파일 경로를 주면 통화마다 단계별 시간·업무 판정·TTS 시간을 JSON 한 줄씩 기록 |
| `TTS_CACHE_DIR` | `runs/tts_cache` | 고정 문구 음성을 저장해 두는 곳. 서버를 다시 켤 때 TTS 를 다시 요청하지 않아 준비가 빠름 |
| `TTS_VOICE` / `TTS_STYLE` | Kore / `Say: ` | 목소리 종류 / 읽어 달라는 지시 (지시가 없으면 Gemini가 실패함) |
| `VERBOSE` | 1 | 0이면 고객 발화를 출력하지 않음 (**정식 측정 때는 0**) |

## 실험: 업무 분류 + 업무별 프롬프트 + 문장별 병렬 TTS
목적: Jev 가 고른 업무 유형을 LLM 에 넘겨 업무마다 프롬프트를 다르게 해서 LLM 응답을 빠르게 하고, TTS 는 문장마다 동시에 만들어 첫 소리를 당긴다.
흐름: 고객이 말을 멈추고 300ms 가 지나면 받아쓰기 중간 결과로 업무 유형을 미리 분류(Jev) → 말 끝 판정(800ms) 뒤 그 업무용 프롬프트로 LLM 호출 → 답을 문장으로 나눠 문장마다 TTS 를 동시에 요청하고 순서대로 이어 붙여 전송.
업무용 프롬프트(`CASE_PROMPT=slim`, 기본): 1단계 공통 규칙(`prompt.md` 의 `# [매뉴얼]` 앞부분) + 2단계 그 업무 절차와 필요한 매뉴얼(`cases/*.md`) + 3단계 통화 정보, 도구는 그 업무 것만(`cases.json` 의 `tools`).
업무를 못 정하면(기타, 확신 부족, 분류 실패) 예전과 같은 전체 프롬프트·전체 도구를 씁니다. **11/21 매뉴얼이 나오면 업무마다 필요한 매뉴얼 부분을 `cases/*.md` 에도 옮겨 적어야 합니다.**

API 비용 없이 먼저 확인: `python eval/fake_apis.py` 를 켜고 `eval/fake_apis.py` 맨 위 설명대로 주소를 바꿔 서버를 켭니다.
실제 API 실험 (저장소 맨 위에서, 순서대로. KO11 은 기본으로 뺍니다):
```powershell
$env:PYTHONUTF8="1"
python eval/bench_case.py --out runs/case1          # 업무 분류 20문장: 키워드 vs Jev
python eval/bench_llm.py --out runs/llm1            # 같은 문장으로 LLM 시간: 전체 프롬프트(full) vs 업무 절차 추가(add) vs 업무용 프롬프트(slim)
python eval/bench_tts.py --out runs/tts2            # TTS 만: 한 번에 / 문장별·키 1개 / 문장별·키 2개 / 앞뒤 2조각 / 같은 글 이중 요청 / 2조각+이중 요청
python eval/experiment.py --out runs/exp2           # 전체 흐름: 녹음 3문항 × 말하기 3가지(stream, split_k2, headtail_hedge2)
python eval/experiment.py --out runs/exp3 --tts stream,stream+early,headtail_hedge2+early   # LLM·TTS 미리 시작(+early) 효과
python eval/experiment.py --out runs/exp4 --tts stream,stream+ack,headtail_hedge2+early+ack # 함수 호출 전 확인 문구(+ack) 효과
python eval/experiment.py --out runs/exp6 --cases data/ko_cases.jsonl --tts stream,stream+ann   # Jev 작업 안내(번호 없이) 끔 vs 켬 (experiment 는 +ann 이 없으면 안내를 끈 채로 잽니다)
python eval/bench_tts.py --out runs/tts4 --presets stream,split3_k3,split3any_k3 --texts T4,T5,T3        # 조회 뒤 답을 3조각까지 나눠 키 3개로
python eval/experiment.py --out runs/exp7 --cases data/ko_cases.jsonl --tts stream+ann,split3_k3+ann,split3any_k3+ann
```

메모 (아직 반영 안 함)
- Decision 을 두 단계로 설계해 답을 더 세분화하는 안: 1단계에서 업무 유형(배송·취소·환불 …)을 고르고, 2단계에서 그 업무 안의 세부 상황(번호를 말했는지, 상태 확인인지 접수인지, 정정했는지 등)을 골라 세부 상황마다 안내 문장·프롬프트·도구를 다르게 쓴다. 지금은 업무 유형(case)과 바로 시작할 작업(action) 두 질문을 한 요청에 나란히 묻는 수준이다.

## 알아 둘 것
- **11/21에 할 일**: 매뉴얼을 `prompt.md`에 붙여 넣기 · 자주 쓰는 고정 문구(`server.py`의 `PHRASES`)를 매뉴얼 문구로 바꾸기 · 도구 정의(`data/demo_tools.json`)와 가짜 장부(`data/demo_mock_data.json`) 교체 · `history` 형식 확인(`server.py`의 `history_messages`).
- **알려진 한계**: Gemini 한도(분당 10회) · 스타터의 `data/audio/demo/meta.json` 값이 녹음과 안 맞음(그래서 직접 녹음한 파일을 쓴다) · 받아쓰기 오인식("교환"→"소환") 보정은 아직 없음.
- **전산실(`mock_tools/`)도 제출물입니다.** 대회 측정 때 상담원 서버와 함께 켜 두어야 하고, 측정 직전에 받는 시험용 장부 파일로 다시 띄웁니다. 겉보기엔 단순하지만 11/21에 가장 많이 바뀌는 부분입니다.
- **제출할 때 뺄 것**: `.env`(키!), `runs/`, `eval/`, `check_apis.py`, `example_server/`.
