#!/usr/bin/env python3
"""Maintenance-only same-binary MTP-head ABBA. Does not stop/restore :8080.

Requires GPUs empty. Starts each isolated arm on loopback:8091, tears it down,
and leaves GPUs free. Temporary credential-bearing configs live in /dev/shm.
"""
import argparse
import copy
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from itertools import combinations
from pathlib import Path

from bench_client import save_json


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def gpu_pids():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15)
    return sorted({int(s.strip()) for s in text.splitlines() if s.strip().isdigit()})


def governors():
    return {str(p):p.read_text().strip() for p in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor'))}


def status(port,key):
    r=urllib.request.Request(f'http://127.0.0.1:{port}/v1/status',headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(r,timeout=5) as f:
        return json.load(f)


def stop(server):
    if server is None:
        return
    try:
        server.terminate()
    except ProcessLookupError:
        pass
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        if not gpu_pids() and server.poll() is not None:
            return
        time.sleep(1)
    # A session created by this runner; never kill unrelated GPU processes.
    try:
        os.killpg(server.pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        if not gpu_pids():
            return
        time.sleep(1)
    raise RuntimeError('GPU processes remain after stopping this arm; inspect manually')


def compare_integrations(out,arms,source_dir):
    available=[r for r in arms if (out/r['tag']/'integration'/'integration.json').exists()]
    comparisons=[]
    for x,y in combinations(available,2):
        # Order cross-arm pairs as disabled -> enabled even for the final A.
        if x['arm']=='on' and y['arm']=='off':
            x,y=y,x
        tag=x['tag']+'-vs-'+y['tag']
        target=out/'integration-comparisons'/tag
        target.mkdir(parents=True,exist_ok=True)
        with (target/'compare.log').open('w') as log:
            subprocess.run([sys.executable,str(source_dir/'integration.py'),'--compare',
                            str(out/x['tag']/'integration'),str(out/y['tag']/'integration'),
                            '--out',str(target)],stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((target/'integration-comparison.json').read_text())
        comparisons.append({'pair':tag,'kind':'cross_flag' if x['arm']!=y['arm'] else x['arm']+'_repeatability',
                            'all_completed_requests_exactly_match':result['all_completed_requests_exactly_match'],
                            'baseline_checks':result['baseline_checks'],'candidate_checks':result['candidate_checks']})
    cross=[r for r in comparisons if r['kind']=='cross_flag']
    save_json(out/'integration-comparison-index.json',{
        'comparisons':comparisons,
        'all_cross_flag_pairs_exact':bool(cross) and all(r['all_completed_requests_exactly_match'] for r in cross),
        'note':'Within-flag pairs help distinguish ordinary concurrency variation from an offload-specific mismatch. One-token logprobs with parallel8 cover the admission path only.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,help='existing base service config, read only')
    p.add_argument('--exe',required=True,help='one rebuilt engine used for BOTH arms')
    p.add_argument('--flag',default='STRATA_MTP_HEAD_GPU',help='environment variable controlling the new path')
    p.add_argument('--off-value',default='-1')
    p.add_argument('--on-value',default='3,5',help='target logical GPU or pair, e.g. 0, 4, or 2,4')
    p.add_argument('--out',required=True,help='new result directory')
    p.add_argument('--server',default='/srv/Strata-0157/serve/server.py')
    p.add_argument('--python',default='/srv/Strata/.venv/bin/python')
    p.add_argument('--port',type=int,default=8091)
    p.add_argument('--arms',default='off,on,on,off')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output-tokens',type=int,default=768)
    p.add_argument('--quick',action='store_true',help='initial off/on correctness smoke only, no 100k performance suite')
    p.add_argument('--suites',default='correctness,performance',help='correctness,performance, optional integration after the main suite')
    p.add_argument('--ready-timeout',type=int,default=1200)
    p.add_argument('--client-timeout',type=int,default=1200)
    a=p.parse_args()
    if a.quick:
        a.arms='off,on'
        a.suites='correctness'+(',integration' if 'integration' in a.suites.split(',') else '')
        a.repeats=1
    suites=a.suites.split(',')
    if set(suites)-{'correctness','performance','integration'} or not set(suites)&{'correctness','performance'}:
        p.error('--suites must include correctness or performance; integration is an optional additional suite')
    client_suites=','.join(s for s in suites if s!='integration')
    if a.port==8080:
        p.error('this runner must not bind the live port 8080')
    arms=a.arms.split(',')
    if set(arms)-{'off','on'}:
        p.error('--arms accepts only off,on')
    if (a.flag,a.off_value,a.on_value)!=('STRATA_MTP_HEAD_GPU','-1','3,5'):
        p.error('This frozen experiment requires MTP head -1 versus 3,5')
    out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        p.error('--out must be new or empty')
    if gpu_pids():
        raise SystemExit('GPUs are occupied; runner did not stop any existing process')
    sock=socket.socket()
    try:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('127.0.0.1',a.port))
    finally:
        sock.close()
    out.mkdir(parents=True,exist_ok=True)
    original=Path(a.config).read_bytes()
    base=json.loads(original)
    if base.get('env',{}).get('STRATA_DECODE_HEAD_GPU')!='2,4':
        p.error('This frozen experiment requires the existing main head on GPU2,4')
    binary=Path(a.exe).resolve()
    binary_hash=sha(binary)
    source_dir=Path(__file__).resolve().parent
    harness_hashes={name:sha(source_dir/name) for name in ('bench_client.py','client.py','run_abba.py','summarize.py','integration.py',
                   'evaluate_final.py','compare_default_off.py','evaluation-plan.json')}
    reference=out/'canonical-append.json'
    start_governors=governors()
    manifest={'created_epoch':time.time(),'sequence':arms,'same_binary_sha256':binary_hash,
              'base_config_sha256':hashlib.sha256(original).hexdigest(),'server_sha256':sha(a.server),
              'flag':a.flag,'off_value':a.off_value,'on_value':a.on_value,'governors_before':start_governors,
              'suites':a.suites,'repeats':a.repeats,'output_tokens':a.output_tokens,
              'harness_sha256':harness_hashes,
              'lifecycle_note':'Requires maintenance; never stops/starts live port8080; only owns its isolated server groups.',
              'arms':[]}
    save_json(out/'manifest.json',manifest)
    server=monitor=None
    config_path=None
    server_log=monitor_log=None
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        for index,arm in enumerate(arms):
            tag=f'{index+1:02d}-{arm}'
            ad=out/tag
            ad.mkdir()
            if gpu_pids():
                raise RuntimeError('Unexpected GPU processes before '+tag)
            if sha(binary)!=binary_hash:
                raise RuntimeError('Rebuilt binary changed during ABBA')
            if any(sha(source_dir/name)!=digest for name,digest in harness_hashes.items()):
                raise RuntimeError('Benchmark scripts changed during ABBA')
            cfg=copy.deepcopy(base)
            cfg.update(exe=str(binary),host='127.0.0.1',port=a.port,log=str(ad/'engine.log'))
            cfg.setdefault('env',{})[a.flag]=a.on_value if arm=='on' else a.off_value
            safe={k:v for k,v in cfg.items() if k!='api_key'}
            # Never export arbitrary secrets present in an inherited environment.
            safe['env']={k:v for k,v in cfg.get('env',{}).items() if k.startswith('STRATA_')}
            stable=copy.deepcopy(safe)
            stable.pop('log',None)
            stable['env'].pop(a.flag,None)
            arm_meta={'tag':tag,'arm':arm,'index':index,'start_epoch':time.time(),'config':safe,
                      'config_except_flag_log_sha256':hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest(),
                      'engine_sha256':sha(binary),'governors_before':governors()}
            save_json(ad/'metadata.json',arm_meta)
            with tempfile.NamedTemporaryFile(mode='w',prefix='strata-mtp-head-abba-',suffix='.json',dir='/dev/shm',delete=False) as f:
                config_path=Path(f.name)
                os.chmod(config_path,0o600)
                json.dump(cfg,f)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(str(x) for x in cfg.get('gpu',range(8))))
            server_log=open(ad/'server.log','w')
            server=subprocess.Popen([a.python,a.server,'--engine','strata','--config',str(config_path),'--port',str(a.port)],
                    cwd=str(Path(a.server).resolve().parents[1]),env=env,stdout=server_log,
                    stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            print('LOADING',tag,'server pid',server.pid,flush=True)
            save_json(out/'active.json',{'arm':tag,'phase':'loading','server_pid':server.pid})
            deadline=time.monotonic()+a.ready_timeout
            while time.monotonic()<deadline:
                if server.poll() is not None:
                    raise RuntimeError(f'{tag} server exited {server.returncode}')
                try:
                    state=status(a.port,cfg['api_key'])
                    if state.get('loaded'):
                        break
                except Exception:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError(tag+' did not become ready')
            arm_meta['ready_epoch']=time.time()
            arm_meta['ready_status']={k:state.get(k) for k in ('loaded','engine','concurrency','activity','vram','machine')}
            save_json(ad/'metadata.json',arm_meta)
            monitor_log=open(ad/'gpu.csv','w')
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv','-lms','500'],stdout=monitor_log,stderr=subprocess.DEVNULL)
            print('READY',tag,'running client',flush=True)
            save_json(out/'active.json',{'arm':tag,'phase':'benchmark','server_pid':server.pid})
            with open(ad/'client.log','w') as log:
                result=subprocess.run([sys.executable,str(source_dir/'client.py'),'--config',str(config_path),
                        '--out',str(ad),'--reference',str(reference),'--port',str(a.port),'--repeats',str(a.repeats),
                        '--output-tokens',str(a.output_tokens),'--suites',client_suites],stdout=log,stderr=subprocess.STDOUT,timeout=a.client_timeout)
            arm_meta['client_returncode']=result.returncode
            integration_returncode=None
            if result.returncode==0 and 'integration' in suites:
                print('INTEGRATION',tag,flush=True)
                save_json(out/'active.json',{'arm':tag,'phase':'integration','server_pid':server.pid})
                with open(ad/'integration.log','w') as log:
                    ir=subprocess.run([sys.executable,str(source_dir/'integration.py'),'--config',str(config_path),
                            '--out',str(ad/'integration'),'--port',str(a.port)],stdout=log,stderr=subprocess.STDOUT,timeout=a.client_timeout)
                integration_returncode=ir.returncode
                arm_meta['integration_returncode']=integration_returncode
            arm_meta['finished_epoch']=time.time()
            arm_meta['governors_after']=governors()
            save_json(ad/'metadata.json',arm_meta)
            manifest['arms'].append({'tag':tag,'arm':arm,'client_returncode':result.returncode,'integration_returncode':integration_returncode})
            save_json(out/'manifest.json',manifest)
            monitor.terminate(); monitor.wait(timeout=10); monitor=None
            monitor_log.close(); monitor_log=None
            stop(server); server=None
            server_log.close(); server_log=None
            config_path.unlink(); config_path=None
            print('FINISHED',tag,'GPU released',flush=True)
            if result.returncode:
                raise RuntimeError(tag+' client failed; preserve evidence and inspect')
            if integration_returncode not in (None,0):
                raise RuntimeError(tag+' integration failed; preserve evidence and inspect')
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        if monitor_log is not None:
            monitor_log.close()
        try:
            stop(server)
        finally:
            if server_log is not None:
                server_log.close()
            if config_path is not None:
                config_path.unlink(missing_ok=True)
            manifest['ended_epoch']=time.time()
            manifest['governors_after']=governors()
            manifest['original_config_unchanged']=Path(a.config).read_bytes()==original
            manifest['remaining_gpu_pids']=gpu_pids()
            save_json(out/'manifest.json',manifest)
            save_json(out/'active.json',{'phase':'stopped','remaining_gpu_pids':manifest['remaining_gpu_pids']})
    subprocess.run([sys.executable,str(source_dir/'summarize.py'),'--root',str(out)],check=True)
    if 'integration' in suites:
        compare_integrations(out,manifest['arms'],source_dir)


if __name__=='__main__':
    main()
