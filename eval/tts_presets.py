# [말하기 설정 묶음] bench_tts.py 와 experiment.py 가 같은 이름으로 같은 설정을 쓰도록 모아 둔 표입니다.
# 값은 서버 환경변수와 같습니다 (TTS_MODE, TTS_MAX_PARTS, TTS_KEYS, TTS_HEDGE, TTS_SPLIT_CLAUSE). 키 2·3개 설정은 .env 에 GEMINI_API_KEY2·3 이 있어야 합니다 (모자라면 있는 키를 돌려 씀).
PRESETS = {
    "stream":          ("한 번에 (지금 서버)",                     {"TTS_MODE": "stream", "TTS_KEYS": "1", "TTS_HEDGE": "1"}),
    "split_k1":        ("문장별 동시 요청, 키 1개",                 {"TTS_MODE": "parallel_ordered", "TTS_MAX_PARTS": "3", "TTS_KEYS": "1", "TTS_HEDGE": "1"}),
    "split_k2":        ("문장별 동시 요청, 키 2개에 나눠",          {"TTS_MODE": "parallel_ordered", "TTS_MAX_PARTS": "3", "TTS_KEYS": "2", "TTS_HEDGE": "1"}),
    "headtail_k2":     ("앞 문장 / 나머지 2조각, 키 2개에 나눠",    {"TTS_MODE": "parallel_ordered", "TTS_MAX_PARTS": "2", "TTS_KEYS": "2", "TTS_HEDGE": "1"}),
    "hedge2":          ("안 쪼갬, 같은 글을 키 2개로 동시 요청",    {"TTS_MODE": "single", "TTS_KEYS": "2", "TTS_HEDGE": "2"}),
    "headtail_hedge2": ("2조각 + 조각마다 키 2개로 동시 요청",      {"TTS_MODE": "parallel_ordered", "TTS_MAX_PARTS": "2", "TTS_KEYS": "2", "TTS_HEDGE": "2"}),
    "split3_k3":       ("최대 3조각(문장 + 쉼표·연결 어미), 키 3개", {"TTS_MODE": "parallel_ordered", "TTS_MAX_PARTS": "3", "TTS_KEYS": "3", "TTS_HEDGE": "1", "TTS_SPLIT_CLAUSE": "1"}),
    "split3any_k3":    ("최대 3조각(+ '은·는·로는' 뒤도), 키 3개",   {"TTS_MODE": "parallel_ordered", "TTS_MAX_PARTS": "3", "TTS_KEYS": "3", "TTS_HEDGE": "1", "TTS_SPLIT_CLAUSE": "2"}),
}


def requests_per_answer(name, parts):
    """이 설정으로 parts 개 문장짜리 답을 읽힐 때 보내는 TTS 요청 수."""
    env = PRESETS[name][1]
    hedge = int(env.get("TTS_HEDGE", 1))
    if env["TTS_MODE"] in ("stream", "single"):
        return hedge
    return min(parts, int(env.get("TTS_MAX_PARTS", 3))) * hedge
