#!/usr/bin/env python3
"""CPU-only full client/control flow, with in-memory HTTP and no socket creation."""
import argparse
import copy
import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import fixture_prompts as fp
import finite_mixed as finite
import stream_client as base
import admission_smoke as smoke
import compare_interleave as compare
import interleave_evidence as evidence
import runner_interleave as runner


class FakeTransport:
    def __init__(self,log,mode,counts):
        self.log,self.mode,self.counts=log,mode,counts;self.lock=threading.Lock();self.requests=0;self.active=0;self.read_id=0
    def append(self,line):
        with self.lock:
            with self.log.open('a') as handle:
                handle.write(line+'\n')
                if line.startswith('strata piped concurrent MTP: end read_id='):
                    read_id=line.split('read_id=',1)[1].split()[0]
                    handle.write(f'strata resident read source: read_id={read_id} staged=1 lookahead_idle=1 dispatch_delta=0\n')
                handle.flush()
    def connection(self,host,port,timeout):
        assert host=='127.0.0.1' and port==8092
        outer=self
        class Sock:
            def __init__(self):self.closed=threading.Event()
            def settimeout(self,value):pass
            def shutdown(self,how):self.closed.set()
        class Connection:
            def __init__(self):self.sock=Sock();self.kind=None;self.finalized=False;self.body=None;self.read=None
            def connect(self):pass
            def request(self,method,path,body=None,headers=None):
                self.path=path;self.body=json.loads(body) if body else None
                if method=='POST' and path=='/v1/chat/completions':
                    self.kind='chat';self.prompt=self.body['messages'][0]['content'];self.tokens=outer.counts[hashlib.sha256(self.prompt.encode()).hexdigest()]
                    self.resident=self.body['max_tokens']>=4096
                    with outer.lock:outer.requests+=1;outer.active+=1;outer.read_id+=1;read_id=outer.read_id
                    if outer.mode=='on' and not self.resident:
                        self.read=read_id;outer.append(f'strata piped concurrent MTP: begin read_id={read_id} from=0 tokens={self.tokens-1} chunk=2048 active_slots=4 stages=8 own_buffers=8 resident_stages=8')
            def finish_read(self):
                if self.read is not None:
                    cancelled=self.sock.closed.is_set()
                    outer.append(f'strata piped concurrent MTP: end read_id={self.read} read_ms=50.0 decode_rows_before_done=20 read_ok={0 if cancelled else 1} pump_ok=1 stopped_at={-1 if cancelled else self.tokens-1} admit_stop_requested=0 yield_slot=-1')
                    if cancelled:outer.append(f'strata serve: prompt {self.tokens} tokens = 0 reused + 8192 of {self.tokens} read in 50 ms (163840.0 tok/s), 0 generated in 0 ms (0.0 tok/s), drafts accepted 0 of 0, 0 checkpoints (cancelled)')
                    self.read=None
            def getresponse(self):
                connection=self
                class Response:
                    status=200
                    def read(self,*args):
                        if connection.path=='/v1/status':
                            with outer.lock:requests,active=outer.requests,outer.active
                            return json.dumps({'loaded':True,'activity':{'requests':requests,'in_flight':active},'concurrency':{'serving':8,'requested':8}}).encode()
                        if connection.path=='/v1/messages/count_tokens':
                            prompt=connection.body['messages'][0]['content'];return json.dumps({'input_tokens':outer.counts[hashlib.sha256(prompt.encode()).hexdigest()]}).encode()
                        raise AssertionError(connection.path)
                    def __iter__(self):
                        delay=.02 if connection.resident else .05
                        count=35 if connection.resident else 2
                        # A long cold read remains observable before the first output.
                        if not connection.resident:
                            connection.sock.closed.wait(.07)
                            if connection.sock.closed.is_set():connection.finish_read();return
                            connection.finish_read()
                        for index in range(count):
                            connection.sock.closed.wait(delay)
                            if connection.sock.closed.is_set():connection.finish_read();return
                            value={'choices':[{'delta':{'content':str(index)+' synthetic answer '}}]}
                            yield b'data: '+json.dumps(value).encode()+b'\n';yield b'\n'
                        completion=connection.body['max_tokens']
                        value={'choices':[{'delta':{},'finish_reason':'length'}],
                               'usage':{'prompt_tokens':connection.tokens,'completion_tokens':completion,'total_tokens':connection.tokens+completion,'prompt_tokens_details':{'cached_tokens':0}},
                               'timings':{'cache_n':0,'prompt_n':connection.tokens,'prompt_ms':70.0,'unapproved_field':'DO_NOT_KEEP'}}
                        yield b'data: '+json.dumps(value).encode()+b'\n';yield b'\n';yield b'data: [DONE]\n';yield b'\n'
                return Response()
            def close(self):
                self.finish_read()
                if self.kind=='chat' and not self.finalized:
                    with outer.lock:outer.active-=1
                    self.finalized=True
        return Connection()


def main():
    evidence.self_test();runner.self_test()
    original_http=http.client.HTTPConnection;original_plan=finite.plan;original_wait=base.Client.wait
    def fast_plan(case):
        p=original_plan(case);p.update(resident_ready_stable_seconds=.02,baseline_seconds=.02,injection_interval_seconds=.04,recovery_seconds=.03,occupancy_sample_seconds=.01);return p
    counts={};frozen=json.loads(Path(__file__).parents[1].joinpath('calibration.json').read_text())
    for row in frozen['fixtures']:counts[row['prompt_sha256']]=row['counted_prompt_tokens']
    for row in smoke.specs():counts[hashlib.sha256(fp.prompt(row['words'],row['fixture']).encode()).hexdigest()]=60000 if row['words']>10000 else 650
    try:
        finite.plan=fast_plan
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);results={}
            for arm in ('off','on'):
                log=root/(arm+'.log');log.write_text('');transport=FakeTransport(log,arm,counts);http.client.HTTPConnection=transport.connection
                cfg=root/(arm+'.json');cfg.write_text(json.dumps({'model_name':frozen['model'],'api_key':'OFFLINE_TEST_SECRET','env':{'STRATA_PIPED_CONCURRENT_MTP':arm=='on' and '1' or '0'}}))
                out=root/('finite-'+arm)
                args=argparse.Namespace(port=8092,exclusive_benchmark=True,config=cfg,calibration=Path(__file__).parents[1]/'calibration.json',out=out,engine_log=log,case='primary')
                assert finite.run(args)==0
                results[arm]=out
            comparison=compare.finite(results['off'],results['on']);assert comparison['all_checks_passed']
            # Mutated complete outputs/counts are rejected, including unchanged request set.
            left=json.loads((results['off']/'results.json').read_text())['records'];right=copy.deepcopy(left)
            right[0]['output_sha256']='wrong';assert not compare.records_equal(left,right)['all_exact']
            right=copy.deepcopy(left);right[0]['usage']['total_tokens']+=1;assert not compare.records_equal(left,right)['all_exact']
            row=frozen['fixtures'][0];good=copy.deepcopy(left[0]);assert all(finite.request_gates(good,row).values()) if good['name']==row['name'] else True
            for key,value in (('prompt_tokens',9999),('completion_tokens',0),('total_tokens',0)):
                bad=copy.deepcopy(good);bad['usage'][key]=value;assert not all(finite.request_gates(bad,row).values())
            # Candidate cancellation smoke exercises a genuinely open mocked reader.
            log=root/'smoke.log';log.write_text('');transport=FakeTransport(log,'on',counts);http.client.HTTPConnection=transport.connection
            args=argparse.Namespace(port=8092,exclusive_benchmark=True,config=root/'on.json',out=root/'smoke',engine_log=log,arm='on')
            assert smoke.run(args)==0
            import cancel_read_proof
            approved=cancel_read_proof.approved_from_results(args.out,log,'on')
            assert len(approved)==1
            gate=evidence.validate('on','admission',log.read_text(),approved_cancelled_read_ids=approved);assert gate['all_checks_passed']
            assert not evidence.validate('on','admission',log.read_text())['all_checks_passed']
            # One damaged byte in the closed own-cancel window invalidates approval.
            original_log=log.read_bytes();log.write_bytes(original_log.replace(b'8192 of',b'8193 of',1))
            try:cancel_read_proof.approved_from_results(args.out,log,'on')
            except RuntimeError:pass
            else:raise AssertionError('Mutated cancelled log accepted')
            log.write_bytes(original_log)
            for path in root.rglob('*.json'):
                if path.name not in ('on.json','off.json'):
                    assert 'OFFLINE_TEST_SECRET' not in path.read_text() and 'unapproved_field' not in path.read_text()
    finally:
        http.client.HTTPConnection=original_http;finite.plan=original_plan;base.Client.wait=original_wait
    print('PASS: fake HTTP finite off/on, matched outputs, final usage/native timings, all phases, output/count mutation rejection, reader-open cancellation/reuse, credential allowlist; no real network/GPU')


if __name__=='__main__':main()
