#!/usr/bin/env python3
"""CPU full-flow incremental cache tests; in-memory HTTP, no sockets/GPU."""
import argparse,copy,hashlib,http.client,json,tempfile,threading,time
from pathlib import Path
import incremental_mixed as inc
import compare_interleave as compare
import stream_client as base

class Fake:
    def __init__(self,log,mode,counts):self.log,self.mode,self.counts=log,mode,counts;self.lock=threading.Lock();self.requests=0;self.active=0;self.nextid=0
    def append(self,s):
        with self.lock:
            with self.log.open('a') as f:f.write(s+'\n')
    def connection(self,host,port,timeout):
        assert host=='127.0.0.1' and port==8092
        outer=self
        class Sock:
            def __init__(self):self.closed=threading.Event()
            def settimeout(self,v):pass
            def shutdown(self,v):self.closed.set()
        class Connection:
            def __init__(self):self.sock=Sock();self.kind=None;self.final=False
            def connect(self):pass
            def request(self,method,path,body=None,headers=None):
                self.path=path;self.body=json.loads(body) if body else None
                if path=='/v1/chat/completions':
                    self.kind='chat';self.prompt=self.body['messages'][0]['content'];self.tokens=outer.counts[hashlib.sha256(self.prompt.encode()).hexdigest()]
                    self.resident=self.body['max_tokens']==4096;self.increment=self.tokens>100000;self.cached=self.tokens-1007 if self.increment else 0
                    with outer.lock:outer.requests+=1;outer.active+=1;outer.nextid+=1;self.id=outer.nextid
                    if self.increment and self.tokens==101000:outer.append(f'strata serve: conversation cache: restored {self.cached} tokens (checkpoint) in 10.0 ms; parked=5 bytes=2000000000')
                    if self.increment and outer.mode=='on':outer.append(f'strata piped concurrent MTP: begin read_id={self.id} from={self.cached} tokens=1000 chunk=8192 active_slots=4 stages=8 own_buffers=8 resident_stages=8')
            def getresponse(self):
                c=self
                class Response:
                    status=200
                    def read(self,*args):
                        if c.path=='/v1/status':
                            with outer.lock:req,active=outer.requests,outer.active
                            return json.dumps({'loaded':True,'activity':{'requests':req,'in_flight':active},'concurrency':{'serving':8,'requested':8}}).encode()
                        if c.path=='/v1/messages/count_tokens':return json.dumps({'input_tokens':outer.counts[hashlib.sha256(c.body['messages'][0]['content'].encode()).hexdigest()]}).encode()
                        raise AssertionError(c.path)
                    def __iter__(self):
                        if not c.resident:
                            time.sleep(.05)
                            if c.increment and outer.mode=='on':
                                outer.append(f'strata piped concurrent MTP: end read_id={c.id} read_ms=50.0 decode_rows_before_done=20 read_ok=1 pump_ok=1 stopped_at=-1 admit_stop_requested=0 yield_slot=-1')
                                outer.append(f'strata resident read source: read_id={c.id} staged=1 lookahead_idle=1 dispatch_delta=0')
                            outer.append(f'strata serve: prompt {c.tokens} tokens = {c.cached} reused + {c.tokens-c.cached} read in 50 ms (1000.0 tok/s), {1 if c.increment else 64} generated in 20 ms (50.0 tok/s), drafts accepted 0 of 0, 1 checkpoints')
                        for i in range(100 if c.resident else 2):
                            c.sock.closed.wait(.01)
                            if c.sock.closed.is_set():return
                            yield b'data: '+json.dumps({'choices':[{'delta':{'content':str(i)+' synthetic content '}}]}).encode()+b'\n';yield b'\n'
                        actual=2815 if c.resident and 'primary-resident-2:' in c.prompt else c.body['max_tokens'];reason='stop' if actual==2815 else 'length'
                        event={'choices':[{'delta':{},'finish_reason':reason}], 'usage':{'prompt_tokens':c.tokens,'completion_tokens':actual,'total_tokens':c.tokens+actual,'prompt_tokens_details':{'cached_tokens':c.cached}},
                               'timings':{'cache_n':c.cached,'prompt_n':c.tokens-c.cached,'prompt_ms':50.,'predicted_n':actual}}
                        yield b'data: '+json.dumps(event).encode()+b'\n';yield b'\n';yield b'data: [DONE]\n';yield b'\n'
                return Response()
            def close(self):
                if self.kind=='chat' and not self.final:
                    with outer.lock:outer.active-=1
                    self.final=True
        return Connection()

def main():
    original_http=http.client.HTTPConnection;original_plan=inc.plan
    def fast():
        p=original_plan();p.update(resident_ready_stable_seconds=.02,baseline_seconds=.02,injection_interval_seconds=.06,recovery_seconds=.03,occupancy_sample_seconds=.01);return p
    calibration=Path(__file__).parents[1]/'calibration.json';frozen=json.loads(calibration.read_text());specs,texts=inc.fixtures(frozen)
    parents=[r for r in specs if r['role'] in ('seed','injected')]
    assert len(specs)==8 and len(texts)==8
    assert not {r['prompt_sha256'] for r in specs}&{r['prompt_sha256'] for r in frozen['fixtures']}
    for before,after in zip(parents,parents[1:]):assert texts[after['fixture']].startswith(texts[before['fixture']]) and after['prefix_parent_sha256']==before['prompt_sha256']
    counts={r['prompt_sha256']:(101000+1000*r['extension_index'] if r['role']=='injected' else r['source_calibrated_prompt_tokens']) for r in specs}
    try:
        inc.plan=fast
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);outputs={}
            for arm in ('off','on'):
                log=root/(arm+'.log');log.write_text('');http.client.HTTPConnection=Fake(log,arm,counts).connection
                cfg=root/(arm+'.json');cfg.write_text(json.dumps({'model_name':frozen['model'],'api_key':'OFFLINE_PRIVATE_KEY','env':{'STRATA_PIPED_CONCURRENT_MTP':'1' if arm=='on' else '0','STRATA_PARK_OVERWRITE':'0'}}))
                a=argparse.Namespace(config=cfg,calibration=calibration,out=root/arm,engine_log=log,port=8092,exclusive_benchmark=True)
                assert inc.run(a)==0;outputs[arm]=a.out
            good=compare.finite(outputs['off'],outputs['on']);assert good['all_checks_passed']
            report=json.loads((outputs['on']/'results.json').read_text());manifest=json.loads((outputs['on']/'manifest.json').read_text())
            row=next(r for r in manifest['fixtures'] if r['name']=='incremental-injected-0');record=next(r for r in report['records'] if r['name']==row['name'])
            assert all(inc.gates(record,row,'on').values())
            for mutation in ('missing_restore','wrong_cache','cold','wrong_from','no_progress','native_mismatch','bad_finish'):
                bad=copy.deepcopy(record)
                if mutation=='missing_restore':bad['admission_read_evidence']['restore_records']=[]
                elif mutation=='wrong_cache':bad['usage']['prompt_tokens_details']['cached_tokens']-=1
                elif mutation=='cold':bad['usage']['prompt_tokens_details']['cached_tokens']=0
                elif mutation=='wrong_from':bad['admission_read_evidence']['reader_pump']['pairs'][0]['from']=0
                elif mutation=='no_progress':bad['admission_read_evidence']['reader_pump']['pairs'][0]['decode_rows_before_done']=0
                elif mutation=='native_mismatch':bad['admission_read_evidence']['native_prompt_records'][0]['read_tokens']+=1
                else:bad['finish_reason']='tool_calls'
                assert not all(inc.gates(bad,row,'on').values()),mutation
            # Cache metadata differences reject a pair even when output and logical usage match.
            original=(outputs['on']/'results.json').read_text();changed=copy.deepcopy(report);changed['records'][0]['cached_tokens']+=1
            (outputs['on']/'results.json').write_text(json.dumps(changed));assert not compare.finite(outputs['off'],outputs['on'])['all_checks_passed'];(outputs['on']/'results.json').write_text(original)
            bounds={k:(v['visible_content']['begin_epoch'],v['visible_content']['end_epoch']) for k,v in report['phases'].items()}
            changed=copy.deepcopy(report['records']);peer=next(r for r in changed if r['role']=='resident');peer['chunks']=[c for c in peer['chunks'] if c['epoch']<bounds['recovery'][1]-.02];peer['end_epoch']=bounds['recovery'][1]+100
            assert not inc.fm.visible_phase_coverage(changed,bounds)['all_checks_passed']
            for path in root.rglob('*.json'):
                if path.name not in ('on.json','off.json'):assert 'OFFLINE_PRIVATE_KEY' not in path.read_text()
    finally:http.client.HTTPConnection=original_http;inc.plan=original_plan
    print('PASS: incremental full fake HTTP off/on, seed-cold and cross-conversation restore, exact actual output/cache parity, normal EOS and real-visible coverage; mismatch/zero-cache/no-restore/reader/native/finish/late-terminal negatives; no real network/GPU')
if __name__=='__main__':main()
