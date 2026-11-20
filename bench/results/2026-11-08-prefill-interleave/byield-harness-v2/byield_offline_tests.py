#!/usr/bin/env python3
"""No-socket integration test for independent BYIELD qualification."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import byield_smoke as test
import offline_tests as previous


class Transport(previous.FakeTransport):
    def __init__(self,*args):super().__init__(*args);self.short=threading.Event();self.long_count=None
    def connection(self,host,port,timeout):
        conn=super().connection(host,port,timeout);request=conn.request;getresponse=conn.getresponse;outer=self
        def wrapped_request(method,path,body=None,headers=None):
            request(method,path,body,headers)
            if conn.kind=='chat' and conn.body['max_tokens']==128:outer.long_count=conn.tokens
            if conn.kind=='chat' and conn.body['max_tokens']==64:outer.short.set()
        def wrapped_response():
            if conn.path=='/v1/telemetry':
                class Telemetry:
                    status=200
                    def read(self,*args):return json.dumps({'work':{'slots':[{'slot':4,'state':'reading','prompt_tokens':outer.long_count,'prompt_read':0}]}}).encode()
                return Telemetry()
            response=getresponse()
            if conn.kind!='chat' or conn.body['max_tokens']!=128:return response
            class Long:
                status=200
                def __iter__(self):
                    assert outer.short.wait(3),'Short request was never submitted'
                    if outer.mode=='on':
                        first=conn.read;conn.read=None
                        outer.append(f'strata piped concurrent MTP: end read_id={first} read_ms=25.0 decode_rows_before_done=20 read_ok=1 pump_ok=1 stopped_at=4096 admit_stop_requested=0 yield_slot=4')
                        outer.append(f'strata batch: the prompt read gives way at 4096 of {conn.tokens} tokens; slot 4 holds it (copied in 1.2 ms)')
                        time.sleep(.20)
                        outer.append(f'strata piped concurrent MTP: begin read_id=100 from=4096 tokens={conn.tokens-1-4096} chunk=2048 active_slots=4 stages=8 own_buffers=8 resident_stages=8')
                        time.sleep(.05)
                        outer.append('strata piped concurrent MTP: end read_id=100 read_ms=50.0 decode_rows_before_done=10 read_ok=1 pump_ok=1 stopped_at=-1 admit_stop_requested=0 yield_slot=-1')
                    yield from response
            return Long()
        conn.request=wrapped_request;conn.getresponse=wrapped_response;return conn


def main():
    test.self_test();original_http=http.client.HTTPConnection;original_post=test.post
    try:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);outputs=[]
            for arm in ('off','on'):
                log=root/(arm+'.log');log.write_text('');counts={};transport=Transport(log,arm,counts);http.client.HTTPConnection=transport.connection
                def fake_count(port,key,path,body,timeout=180):
                    assert path=='/v1/messages/count_tokens';text=body['messages'][0]['content'];count=100+len(text.split('\n')[1].split())
                    counts[hashlib.sha256(text.encode()).hexdigest()]=count;return {'input_tokens':count}
                test.post=fake_count;cfg=root/(arm+'.json');cfg.write_text(json.dumps({'api_key':'OFFLINE_KEY_NEVER_EXPORT','env':{'STRATA_PIPED_CONCURRENT_MTP':'1' if arm=='on' else '0'}}))
                out=root/('result-'+arm);outputs.append(out)
                args=argparse.Namespace(port=8092,exclusive_benchmark=True,config=cfg,out=out,engine_log=log,arm=arm)
                assert test.run(args)==0
            comparison=test.compare(*outputs);assert comparison['all_checks_passed']
            for path in root.rglob('*.json'):
                if path.name not in ('off.json','on.json'):assert 'OFFLINE_KEY_NEVER_EXPORT' not in path.read_text()
            original=json.loads((outputs[1]/'results.json').read_text());original['records'][0]['output_sha256']='corrupted'
            (outputs[1]/'results.json').write_text(json.dumps(original));assert not test.compare(*outputs)['all_checks_passed']
    finally:http.client.HTTPConnection=original_http;test.post=original_post
    print('PASS: full BYIELD off/on fake HTTP, real client wait/thread lifecycle, count calibration, short dispatch, park/resume, exact normal hashes/counts, altered-output rejection, no real HTTP/GPU')


if __name__=='__main__':main()
