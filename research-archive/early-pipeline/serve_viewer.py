"""Loopback-only static viewer with byte ranges for local video and PLY."""
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import re,os
ROOT=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/viewer')
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kw):super().__init__(*args,directory=str(ROOT),**kw)
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy','same-origin')
        self.send_header('Cross-Origin-Embedder-Policy','require-corp')
        self.send_header('Cache-Control','no-cache')
        self.send_header('Accept-Ranges','bytes')
        super().end_headers()
    def send_head(self):
        self.remaining=None
        path=Path(self.translate_path(self.path))
        interval=self.headers.get('Range')
        if not interval or not path.is_file():return super().send_head()
        n=path.stat().st_size;m=re.fullmatch(r'bytes=(\d*)-(\d*)',interval)
        if not m or (not m[1] and not m[2]):self.send_error(416);return
        lo=int(m[1]) if m[1] else max(0,n-int(m[2]));hi=min(n-1,int(m[2])) if m[1] and m[2] else n-1
        if lo>hi or lo>=n:
            self.send_response(416);self.send_header('Content-Range',f'bytes */{n}');self.send_header('Content-Length','0');self.end_headers();return
        f=path.open('rb');f.seek(lo);self.remaining=hi-lo+1
        self.send_response(206);self.send_header('Content-Type',self.guess_type(str(path)));self.send_header('Content-Range',f'bytes {lo}-{hi}/{n}');self.send_header('Content-Length',str(self.remaining));self.end_headers();return f
    def copyfile(self,source,output):
        if self.remaining is None:return super().copyfile(source,output)
        left=self.remaining
        while left:
            chunk=source.read(min(left,1024*1024))
            if not chunk:break
            output.write(chunk);left-=len(chunk)
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',8870),Handler).serve_forever()
