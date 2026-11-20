#!/usr/bin/env python3
"""CPU full-flow incremental cache tests; in-memory HTTP, no sockets/GPU."""
import argparse,copy,hashlib,http.client,json,tempfile,threading,time
from pathlib import Path
import incremental_mixed as inc
import compare_interleave as compare
import stream_client as base

class Fake:
    def __init__(self,log,mode,counts):self.log,self.mode,self.counts=log,mode,counts;self.lock=threading.Lock();self.requests=0;self.active=0;self.nextid=0;self.bodies=[]
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
                    self.kind='chat';self.prompt=self.body['messages'][-1]['content'];self.tokens=outer.counts[hashlib.sha256(self.prompt.encode()).hexdigest()]
                    self.resident=self.body['max_tokens']==4096;self.increment=self.tokens>100000;self.cached=self.tokens-1007 if self.increment else 0
                    with outer.lock:outer.requests+=1;outer.active+=1;outer.nextid+=1;self.id=outer.nextid;outer.bodies.append(copy.deepcopy(self.body))
                    if outer.mode=='on' and not self.resident and not self.increment:outer.append(f'strata piped concurrent MTP: fallback from=0 tokens={self.tokens-7} reason=no_active_slots')
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
                        if c.path=='/v1/messages/count_tokens':return json.dumps({'input_tokens':outer.counts[hashlib.sha256(c.body['messages'][-1]['content'].encode()).hexdigest()]}).encode()
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
                            content=(str(i)+' synthetic content ') if c.resident else ('SYNTHETIC_REPLY_'+str((len(c.body['messages'])-1)//2)+'_'+str(i)+' ')
                            yield b'data: '+json.dumps({'choices':[{'delta':{'content':content}}]}).encode()+b'\n';yield b'\n'
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
    buffer=bytearray();inc.cc.append_capture(buffer,'空闲GPU '.encode());assert buffer.decode()=='空闲GPU '
    buffer=bytearray(b'x'*inc.cc.CAPTURE_LIMIT_BYTES)
    try:inc.cc.append_capture(buffer,b'x')
    except RuntimeError:pass
    else:raise AssertionError('Unbounded assistant capture accepted')
    assert len(buffer)==inc.cc.CAPTURE_LIMIT_BYTES
    original_http=http.client.HTTPConnection;original_plan=inc.plan
    def fast():
        p=original_plan();p.update(resident_ready_stable_seconds=.02,baseline_seconds=.02,injection_interval_seconds=.06,recovery_seconds=.03,occupancy_sample_seconds=.01);return p
    calibration=Path(__file__).parents[1]/'calibration.json';frozen=json.loads(calibration.read_text());specs,texts=inc.fixtures(frozen)
    parents=[r for r in specs if r['role'] in ('seed','injected')]
    assert len(specs)==8 and len(texts)==8
    assert not {r['prompt_sha256'] for r in specs}&{r['prompt_sha256'] for r in frozen['fixtures']}
    for row in parents[1:]:assert len(texts[row['fixture']])<10000 and not texts[row['fixture']].startswith(texts[parents[0]['fixture']])
    counts={r['prompt_sha256']:(101000+1000*r['extension_index'] if r['role']=='injected' else r['source_calibrated_prompt_tokens']) for r in specs}
    try:
        inc.plan=fast
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);outputs={}
            tokenizer=root/'tokenizer';tokenizer.mkdir();(tokenizer/'chat_template.jinja').write_bytes(Path(inc.__file__).with_name('chat_template-active.jinja').read_bytes())
            for arm in ('off','on'):
                log=root/(arm+'.log');log.write_text('');transport=Fake(log,arm,counts);http.client.HTTPConnection=transport.connection
                cfg=root/(arm+'.json');cfg.write_text(json.dumps({'model_name':frozen['model'],'tokenizer':str(tokenizer),'api_key':'OFFLINE_PRIVATE_KEY','env':{'STRATA_PIPED_CONCURRENT_MTP':'1' if arm=='on' else '0','STRATA_PARK_OVERWRITE':'0'}}))
                a=argparse.Namespace(config=cfg,calibration=calibration,out=root/arm,engine_log=log,port=8092,exclusive_benchmark=True)
                assert inc.run(a)==0;outputs[arm]=a.out
                conversation=[b for b in transport.bodies if b['max_tokens']==64]
                assert [len(b['messages']) for b in conversation]==[1,3,5,7]
                for index,(earlier,later) in enumerate(zip(conversation,conversation[1:])):
                    assert later['messages'][:-2]==earlier['messages'] and later['messages'][-2]['role']=='assistant' and later['messages'][-1]['role']=='user'
                    assert later['messages'][-2]['content']=='SYNTHETIC_REPLY_'+str(index)+'_0 SYNTHETIC_REPLY_'+str(index)+'_1 '  # preserve exact trailing whitespace in memory
            good=compare.finite(outputs['off'],outputs['on']);assert good['all_checks_passed']
            report=json.loads((outputs['on']/'results.json').read_text());manifest=json.loads((outputs['on']/'manifest.json').read_text())
            row=next(r for r in manifest['fixtures'] if r['name']=='incremental-injected-0');record=next(r for r in report['records'] if r['name']==row['name'])
            assert all(inc.gates(record,row,'on').values())
            # A larger serialized tail is valid only within the proven preceding turn checkpoint.
            expanded=copy.deepcopy(record);expanded_row=copy.deepcopy(row);expanded_row['counted_prompt_tokens']+=600
            expanded['usage']['prompt_tokens']+=600;expanded['usage']['total_tokens']+=600;expanded['native_timings']['prompt_n']+=600
            expanded['admission_read_evidence']['native_prompt_records'][0]['prompt_tokens']+=600;expanded['admission_read_evidence']['native_prompt_records'][0]['read_tokens']+=600
            expanded['admission_read_evidence']['reader_pump']['pairs'][0]['tokens']+=600
            assert expanded['native_timings']['prompt_n']>1300 and all(inc.gates(expanded,expanded_row,'on').values())
            for mutation in ('missing_restore','wrong_cache','cold','wrong_from','no_progress','native_mismatch','bad_finish','boundary_mismatch','wrong_assistant_chain','periodic_checkpoint_replay'):
                bad=copy.deepcopy(record)
                if mutation=='missing_restore':bad['admission_read_evidence']['restore_records']=[]
                elif mutation=='wrong_cache':bad['usage']['prompt_tokens_details']['cached_tokens']-=1
                elif mutation=='cold':bad['usage']['prompt_tokens_details']['cached_tokens']=0
                elif mutation=='wrong_from':bad['admission_read_evidence']['reader_pump']['pairs'][0]['from']=0
                elif mutation=='no_progress':bad['admission_read_evidence']['reader_pump']['pairs'][0]['decode_rows_before_done']=0
                elif mutation=='native_mismatch':bad['admission_read_evidence']['native_prompt_records'][0]['read_tokens']+=1
                elif mutation=='boundary_mismatch':bad['admission_read_evidence']['reader_pump']['pairs'][0]['tokens']-=1
                elif mutation=='periodic_checkpoint_replay':
                    bad['usage']['prompt_tokens_details']['cached_tokens']=98304;bad['native_timings']['cache_n']=98304;bad['native_timings']['prompt_n']=bad['usage']['prompt_tokens']-98304
                    bad['admission_read_evidence']['native_prompt_records'][0]['reused_tokens']=98304;bad['admission_read_evidence']['native_prompt_records'][0]['read_tokens']=bad['usage']['prompt_tokens']-98304
                    bad['admission_read_evidence']['reader_pump']['pairs'][0]['from']=98304;bad['admission_read_evidence']['reader_pump']['pairs'][0]['tokens']=bad['usage']['prompt_tokens']-98304-7
                elif mutation=='wrong_assistant_chain':bad['continuation_dependency']['previous_visible_sha256']='wrong'
                else:bad['finish_reason']='tool_calls'
                assert not all(inc.gates(bad,row,'on').values()),mutation
            assert inc.conversation_chain_valid(report['records'])
            chain=copy.deepcopy(report['records']);next(r for r in chain if r['name']=='incremental-injected-1')['continuation_dependency']['previous_visible_sha256']='wrong'
            assert not inc.conversation_chain_valid(chain)
            # Cache metadata differences reject a pair even when output and logical usage match.
            original=(outputs['on']/'results.json').read_text();changed=copy.deepcopy(report);changed['records'][0]['cached_tokens']+=1
            (outputs['on']/'results.json').write_text(json.dumps(changed));assert not compare.finite(outputs['off'],outputs['on'])['all_checks_passed'];(outputs['on']/'results.json').write_text(original)
            bounds={k:(v['visible_content']['begin_epoch'],v['visible_content']['end_epoch']) for k,v in report['phases'].items()}
            changed=copy.deepcopy(report['records']);peer=next(r for r in changed if r['role']=='resident');peer['chunks']=[c for c in peer['chunks'] if c['epoch']<bounds['recovery'][1]-.02];peer['end_epoch']=bounds['recovery'][1]+100
            assert not inc.fm.visible_phase_coverage(changed,bounds)['all_checks_passed']
            for path in [*root.rglob('*.json'),*root.rglob('*.jsonl')]:
                if path.name not in ('on.json','off.json'):assert 'OFFLINE_PRIVATE_KEY' not in path.read_text() and 'SYNTHETIC_REPLY_' not in path.read_text()
    finally:http.client.HTTPConnection=original_http;inc.plan=original_plan
    print('PASS: true-conversation incremental full fake HTTP off/on, seed-cold and cross-conversation restore, exact actual output/cache parity, normal EOS and real-visible coverage; mismatch/zero-cache/no-restore/reader/native/finish/late-terminal negatives; no real network/GPU')
if __name__=='__main__':main()
