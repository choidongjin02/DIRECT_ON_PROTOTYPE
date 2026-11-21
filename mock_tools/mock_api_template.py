# [가짜 주문 전산실] 상담원이 "이 주문 어디쯤이에요?" 하고 물어보러 오는 간단한 HTTP 서버입니다 (8100번).
# 왜 있나: 진짜 DB 없이 정해진 질문엔 정해진 답을 주고, 누가 뭘 물었는지 일지(/calls)를 남겨 채점에 쓰려고요.
# 도구 목록 data/demo_tools.json, 장부 data/demo_mock_data.json 을 읽고 lookup() 이 장부를 찾아 답해요.
"""DIRECT:ON pre-release: Mock API skeleton. Business logic is deliberately omitted."""
import argparse, json, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
TOOLS = json.loads((ROOT / 'data' / 'demo_tools.json').read_text(encoding='utf-8'))
MOCK_DATA = json.loads((ROOT / 'data' / 'demo_mock_data.json').read_text(encoding='utf-8'))
CALLS = []
LOCK = threading.Lock()

def lookup(name, args):
    """MOCK_DATA[name]에서 arguments가 args와 똑같은 항목을 찾아 result를 돌려준다.
    return (response_json_dict, matched_bool)
    """
    for entry in MOCK_DATA.get(name, []):
        if entry["arguments"] == args:
            return entry["result"], True
    return {"status": "not_found", "message": "일치하는 데이터 없음"}, False

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass
    def send_obj(self, obj, code=200):
        buf=json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(buf)))
        self.end_headers()
        self.wfile.write(buf)
    def do_GET(self):
        u=urlparse(self.path)
        if u.path=='/tools':return self.send_obj([{'type':'function','function':x} for x in TOOLS])
        if u.path=='/calls':
            sid=parse_qs(u.query).get('session_id',[None])[0]
            with LOCK: rows=[c for c in CALLS if sid is None or c['session_id']==sid]
            return self.send_obj(rows)
        return self.send_obj({'error':'not found'},404)
    def do_POST(self):
        name=urlparse(self.path).path.removeprefix('/tools/')
        if name not in {x['name'] for x in TOOLS}:return self.send_obj({'error':'unknown function'},404)
        try:
            args=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
        except (ValueError,TypeError):return self.send_obj({'error':'invalid JSON'},400)
        result, matched=lookup(name,args)
        with LOCK:
            CALLS.append({'session_id':self.headers.get('X-Session-Id'), 'name':name,
                          'arguments':args, 'matched':bool(matched), 't_response':time.time()})
        return self.send_obj(result)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8100);a=p.parse_args()
    print(f'DEMO mock API skeleton on http://localhost:{a.port} (lookup TODO)',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
