#!/usr/bin/env python3
"""Read-only deployed-engine verification; only local audit receipts are written.

Before restart: python3 verify_piped_deployment.py cursor
After restart:  python3 verify_piped_deployment.py verify
Offline only:  python3 verify_piped_deployment.py self-test

Default action is verify. No process/GPU/service/config mutations. No generation
requests. Status uses the configured API key but neither credentials, raw process
environment, command lines nor raw startup logs are saved or printed.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
import urllib.request

sys.dont_write_bytecode = True
ROOT = Path('/srv/bench100k-20261107-piped-target-head')
CONFIG = Path('/srv/Strata/strata-orca-q5_k_m.57.json')
SERVER = Path('/srv/Strata-0157/serve/server.py')
RELEASE = Path('/srv/Strata-0157/engine/releases/piped-decode-20261107')
PARSER = ROOT/'p2-harness/runner_async_sample.py'
PARSER_SHA = '97db26740aae5b10f65b4bd0d0aa564eda9f578eb4f8bd7171631a20521d8fa6'
BASE_CONFIG_SHA = 'f5376cc83215fc54b46d20094373f20a41b93daa8bf07262d2ac67dc13297220'
SERVER_SHA = '40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9'
LIBRARY_SHA = 'fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661'
SELECTED = ('STRATA_PIPED_TARGET_HEAD', 'STRATA_PIPED_ASYNC_SAMPLE')
BASE_FLAGS = {'STRATA_BATCH_MTP':'1', 'STRATA_PIPED_MTP':'1', 'STRATA_BATCH_SELF_COMMIT':'0',
              'STRATA_PIPED_MTP_HEAD':'1', 'STRATA_DECODE_HEAD_GPU':'2,4', 'STRATA_MTP_HEAD_GPU':'3,5',
              'STRATA_PW_WITH_BATCH':'1', 'STRATA_PREFILL_ZC':'1', 'STRATA_SPLIT_CHUNK_MAX':'8192',
              'STRATA_SPLIT_MTP_BATCH':'1'}
DISABLED = ('STRATA_PIPED_READ_PIECE', 'STRATA_PIPED_READ_DECODE_ONLY', 'STRATA_PIPED_DECODE_MIN_MS',
            'STRATA_PIPED_DECODE_MAX_MS', 'STRATA_PIPED_CONCURRENT', 'STRATA_PIPED_MTP_OBSERVE', 'STRATA_MTP_PROFILE')


def require(ok, why):
    if not ok: raise RuntimeError(why)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    require(not os.path.lexists(path), 'Audit destination already exists; use a new --out or --cursor-file')
    data=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            os.fchmod(handle.fileno(),0o644);handle.write(data);handle.flush();os.fsync(handle.fileno())
        # Hard-link is atomic and refuses an existing destination instead of overwriting another receipt.
        os.link(name,path)
    finally:Path(name).unlink(missing_ok=True)


def no_secret(value,key):
    require(bool(key) and key not in json.dumps(value,ensure_ascii=False), 'Audit contains authentication material; not writing it')


def check_disabled(env):
    require(all(env.get(k) in (None,'','0','0.0',0) for k in DISABLED), 'A rejected read guard or observation/profile switch is enabled')


def check_preserved(old,cfg,flags):
    require(set(flags)==set(SELECTED) and all(v in ('0','1') for v in flags.values()), 'Receipt selected flags are invalid')
    a=copy.deepcopy(old);b=copy.deepcopy(cfg)
    a.pop('exe',None);b.pop('exe',None)
    for k in SELECTED:a['env'].pop(k,None);b['env'].pop(k,None)
    require(a==b, 'A configuration field outside exe/two selected flags differs from the private original')
    require(isinstance(cfg.get('api_key'),str) and bool(cfg['api_key']) and cfg['api_key']==old.get('api_key'), 'API authentication differs')
    require(cfg.get('parallel')==8 and cfg.get('layer_split')=='6,12,18,24,30,36,42', 'Slots/layer split differs')
    require(cfg.get('reasoning_budget_engine_cap') is True, 'Reasoning budget cap differs')
    require(cfg.get('port',8080)==8080, 'Unexpected production port')
    require(cfg.get('exe')==str(RELEASE/'strata'), 'Configuration executable is not the selected release')
    require(all(str(cfg['env'].get(k))==v for k,v in {**BASE_FLAGS,**flags}.items()), 'Configuration baseline or selected flags differ')
    check_disabled(cfg['env'])


def installed_inputs():
    receipt_path=ROOT/'deployment-receipt.json';receipt=load(receipt_path);cfg=load(CONFIG)
    require(receipt.get('mode')=='INSTALLED_SERVICE_NOT_STARTED', 'Unrecognized installation receipt')
    require(receipt.get('release')==str(RELEASE), 'Receipt release path differs')
    expected=receipt['installed_sha256']
    require(set(expected)=={str(CONFIG),str(SERVER),str(RELEASE/'strata'),str(RELEASE/'libstrata_kq.so')}, 'Receipt installed-file set differs')
    require(expected[str(SERVER)]==SERVER_SHA and expected[str(RELEASE/'libstrata_kq.so')]==LIBRARY_SHA, 'Frozen server/library differs')
    require(re.fullmatch('[0-9a-f]{64}',expected[str(RELEASE/'strata')]) is not None, 'Invalid selected engine digest')
    require(all(sha(Path(path))==digest for path,digest in expected.items()), 'Installed file bytes differ from deployment receipt')
    require(stat.S_IMODE(CONFIG.stat().st_mode)==0o600, 'Active configuration permissions are not 0600')
    backup=Path(receipt['backup_directory'])
    require(backup.is_absolute() and backup.parent==ROOT and backup.name.startswith('rollback-piped-decode-') and not backup.is_symlink(), 'Private backup path differs')
    require(stat.S_IMODE(backup.stat().st_mode)==0o700, 'Backup directory permissions are not 0700')
    old_path=backup/CONFIG.name
    require(stat.S_IMODE(old_path.stat().st_mode)==0o600 and sha(old_path)==BASE_CONFIG_SHA, 'Private original configuration differs')
    require(sha(backup/SERVER.name)==SERVER_SHA, 'Private original server differs')
    require(receipt['original_sha256']=={str(CONFIG):BASE_CONFIG_SHA,str(SERVER):SERVER_SHA}, 'Receipt original hashes differ')
    flags=receipt['flags'];check_preserved(load(old_path),cfg,flags)
    return cfg,receipt,sha(receipt_path)


def status(cfg):
    req=urllib.request.Request('http://127.0.0.1:8080/v1/status',headers={'Authorization':'Bearer '+cfg['api_key']})
    with urllib.request.urlopen(req,timeout=8) as response:
        require(response.status==200, 'Authenticated status did not return 200')
        data=json.load(response)
    require(data.get('loaded') is True and data.get('concurrency')=={'serving':8,'requested':8}, 'Service has not loaded eight serving slots')
    return {'loaded':True,'concurrency':{'serving':8,'requested':8},
            'activity':{k:v for k,v in (data.get('activity') or {}).items() if k in ('requests','in_flight','last_request_at') and (v is None or isinstance(v,(int,float)))}}


def boot_id():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def uptime():return float(Path('/proc/uptime').read_text().split()[0])


def mark_log(log):
    s=log.stat() if log.exists() else None;offset=s.st_size if s else 0
    if s is None:
        return {'time_epoch':time.time(),'boot_id':boot_id(),'uptime_s':uptime(),'log':str(log),'device':None,'inode':None,
                'offset':0,'anchor_start':0,'anchor_sha256':hashlib.sha256(b'').hexdigest()}
    begin=max(0,offset-4096)
    with log.open('rb') as h:
        h.seek(begin);tail=h.read(offset-begin)
    return {'time_epoch':time.time(),'boot_id':boot_id(),'uptime_s':uptime(),'log':str(log),
            'device':s.st_dev,'inode':s.st_ino,'offset':offset,'anchor_start':begin,
            'anchor_sha256':hashlib.sha256(tail).hexdigest()}


def log_since(log,mark):
    require(mark['log']==str(log), 'Log path differs from cursor')
    s=log.stat();same=(s.st_dev,s.st_ino)==(mark['device'],mark['inode'])
    offset=0;reason='rotated_or_new_log'
    with log.open('rb') as h:
        if same and s.st_size>=mark['offset']:
            h.seek(mark['anchor_start']);anchor=h.read(mark['offset']-mark['anchor_start'])
            if hashlib.sha256(anchor).hexdigest()==mark['anchor_sha256']:
                offset=mark['offset'];reason='same_log_validated_anchor'
            else:reason='same_inode_copy_truncated_or_rewritten'
        elif same:reason='same_inode_truncated'
        require(s.st_mtime>=mark['time_epoch']-1, 'Startup log is older than the cursor')
        h.seek(offset);text=h.read().decode(errors='replace')
    return text,{'offset':offset,'end_offset':s.st_size,'device':s.st_dev,'inode':s.st_ino,'selection':reason}


def load_parser(path):
    require(sha(path)==PARSER_SHA,'Frozen startup parser SHA256 differs')
    spec=importlib.util.spec_from_file_location('frozen_piped_async_runner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def facts_from_text(parser,text,flags):
    class Text:
        def read_text(self,errors='strict'):return text
    facts=parser.startup_facts(Text())
    require(facts['all_experts_resident'] and facts['all_eight_have_3_80_gib_own_buffers'] and
            facts['batch_slot_devices']==list(range(8)) and facts['piped_mtp_startup_enabled'], 'Fresh startup slot/residency/prefill gates failed')
    require(facts['shared_head_workspaces']==[[8,3,5,7]], 'Fresh baseline MTP shared-head marker differs')
    require(facts['target_head_startup']==([[2,4,7]] if flags[SELECTED[0]]=='1' else []), 'Fresh selected P1 startup marker differs')
    require(facts['async_sample_startup']==([7] if flags[SELECTED[1]]=='1' else []), 'Fresh selected P2 startup marker differs')
    require(not re.search(r'^strata serve: STRATA_PIPED_MTP_OBSERVE:',text,re.M), 'Unexpected P3 observation startup marker')
    return facts


def process_info(proc):
    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':int(proc.name),'ppid':int(fields[1]),'start_ticks':int(fields[19])}


def actual_engine(cfg,receipt,mark):
    require(boot_id()==mark['boot_id'], 'Host rebooted since startup cursor; record a new cursor')
    binary=RELEASE/'strata';library=RELEASE/'libstrata_kq.so';expected=receipt['installed_sha256'];found=[]
    safe_flags={**BASE_FLAGS,**receipt['flags']};hz=os.sysconf('SC_CLK_TCK')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if (proc/'exe').resolve(strict=True)!=binary:continue
            info=process_info(proc)
            require(info['start_ticks']/hz>=mark['uptime_s']-.02,'Installed engine predates cursor; fresh restart not established')
            env=dict(x.split('=',1) for x in (proc/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
            require(all(env.get(k)==str(v) for k,v in cfg['env'].items()),'Actual runtime differs from configured environment')
            require(all(env.get(k)==v for k,v in safe_flags.items()),'Actual baseline or selected engine flags differ')
            check_disabled(env)
            paths={line.split(None,5)[5].strip() for line in (proc/'maps').read_text().splitlines() if len(line.split(None,5))==6}
            libraries=sorted(p for p in paths if 'libstrata_kq.so' in Path(p).name)
            require(libraries==[str(library)], 'Actual mapped KQ library differs or is deleted')
            require(sha(proc/'exe')==expected[str(binary)], 'Running executable image SHA256 differs')
            require(process_info(proc)==info,'Engine PID changed during verification')
            found.append({**info,'exe':str(binary),'engine_sha256':expected[str(binary)],
                          'library_sha256':sha(library),'library_paths':libraries,'fresh_after_cursor':True,
                          'all_configured_env_equal':True,'disabled_guard_and_observation_flags':{k:env.get(k) for k in DISABLED},
                          'flags':{k:env.get(k) for k in safe_flags}})
        except (FileNotFoundError,ProcessLookupError):continue
    require(len(found)==1,'Expected exactly one installed engine process')
    return found


def verify(cursor,out):
    cfg,receipt,receipt_sha=installed_inputs();mark=load(cursor);processes=actual_engine(cfg,receipt,mark)
    safe_status=status(cfg);parser=load_parser(PARSER);text,span=log_since(Path(cfg['log']),mark)
    facts=facts_from_text(parser,text,receipt['flags'])
    require(sha(CONFIG)==receipt['installed_sha256'][str(CONFIG)] and sha(SERVER)==SERVER_SHA,'Configuration/server changed during verification')
    require(process_info(Path('/proc')/str(processes[0]['pid']))['start_ticks']==processes[0]['start_ticks'],'Engine changed during verification')
    result={'time_epoch':time.time(),'all_checks_passed':True,'scope':'Read-only startup/resource/identity acceptance; no generation or performance claim.',
            'deployment_receipt_sha256':receipt_sha,'cursor_sha256':sha(cursor),'parser_sha256':PARSER_SHA,
            'status':safe_status,'config_sha256':sha(CONFIG),'server_sha256':SERVER_SHA,'actual_engine':processes,
            'startup_facts':facts,'fresh_log_span':span,'api_key_present_and_preserved':True,
            'layer_split_and_all_other_config_preserved':True,'reasoning_budget_engine_cap':True,
            'clock_and_power_not_modified':True,'service_mutations':0}
    no_secret(result,cfg['api_key']);save(out,result);print(json.dumps(result,indent=2))


def self_test():
    parser=load_parser(Path(__file__).resolve().parent/'p2-harness/runner_async_sample.py')
    text='pre-filled 3072 of 3072 slots from the profile\n'
    text+=''.join(f'CUDA{i} keeps its own prompt buffers (3.80 GiB of 7.00 GiB free): no loan\n8 slot sessions on CUDA{i}\n' for i in range(8))
    text+=''.join(f'layer split: CUDA{i} runs layers 1-6, expert cache 3072 slots (1.00 GiB), 3072 of its 3072 profiled pairs\n' for i in range(1,8))
    text+='STRATA_PIPED_MTP: 8 pipelined groups of 1 slot draft up to 1 MTP proposal\n'
    text+='STRATA_PIPED_MTP_HEAD: 8 private slot workspaces, CUDA3+5, 1 shared weight bank, source CUDA7\n'
    p1='STRATA_PIPED_TARGET_HEAD: CUDA2+4, source CUDA7, full logits return before original slot sampling; prefill placement unchanged\n'
    p2='strata serve: STRATA_PIPED_ASYNC_SAMPLE: source CUDA7, event-polled slot sampling with persistent history\n'
    for a in ('0','1'):
        for b in ('0','1'):facts_from_text(parser,text+(p1 if a=='1' else '')+(p2 if b=='1' else ''),dict(zip(SELECTED,(a,b))))
    flags=dict(zip(SELECTED,('1','1')))
    rejected=0
    for bad in (text+p1,text+p2,text+p1+p2+text):
        try:facts_from_text(parser,bad,flags)
        except RuntimeError:rejected+=1
    require(rejected==3,'Missing/duplicate startup markers were accepted')
    for key in DISABLED:
        try:check_disabled({key:'1'})
        except RuntimeError:pass
        else:raise RuntimeError('Enabled guard was accepted')
    old={'exe':'/old/strata','api_key':'OFFLINE_KEY','parallel':8,'port':8080,
         'layer_split':'6,12,18,24,30,36,42','reasoning_budget_engine_cap':True,'env':dict(BASE_FLAGS)}
    cfg=copy.deepcopy(old);cfg['exe']=str(RELEASE/'strata');cfg['env'].update(flags)
    check_preserved(old,cfg,flags)
    altered=copy.deepcopy(cfg);altered['env']['STRATA_SPLIT_CHUNK_MAX']='4096'
    try:check_preserved(old,altered,flags)
    except RuntimeError:pass
    else:raise RuntimeError('Changed prefill environment was accepted')
    with tempfile.TemporaryDirectory(prefix='verify-piped-offline-') as tmp:
        root=Path(tmp);log=root/'engine.log';log.write_bytes(b'old\n');offset=log.stat().st_size
        mark={'log':str(log),'device':log.stat().st_dev,'inode':log.stat().st_ino,'offset':offset,
              'anchor_start':0,'anchor_sha256':hashlib.sha256(b'old\n').hexdigest(),'time_epoch':time.time()-1}
        with log.open('ab') as h:h.write(b'fresh\n')
        fresh,span=log_since(log,mark);require(fresh=='fresh\n' and span['offset']==offset,'Cursor did not exclude old log')
        log.write_bytes(b'new-start\n');fresh,span=log_since(log,mark)
        require(fresh=='new-start\n' and span['offset']==0,'Copy-truncate was not detected')
        try:no_secret({'bad':'OFFLINE_KEY'},'OFFLINE_KEY')
        except RuntimeError:pass
        else:raise RuntimeError('Credential leak check failed')
        save(root/'receipt.json',{'safe':True})
        try:save(root/'receipt.json',{'safe':False})
        except RuntimeError:pass
        else:raise RuntimeError('Existing receipt overwritten')
    print(json.dumps({'mode':'OFFLINE_SELF_TEST','passed':True,'checks':['four_selected_flag_combinations','missing_and_duplicate_markers_rejected',
        'enabled_guards_rejected','protected_config_and_api_key_preserved','old_log_excluded','copy_truncate_detected','credential_output_rejected','receipt_overwrite_rejected'],
        'remote_requests':0,'gpu_operations':0,'service_mutations':0},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action',nargs='?',choices=('cursor','verify','self-test'),default='verify')
    p.add_argument('--cursor-file',type=Path,default=ROOT/'deployment-log-cursor.json')
    p.add_argument('--out',type=Path,default=ROOT/'deployment-verified.json')
    a=p.parse_args()
    if a.action=='self-test':self_test();return
    if a.action=='cursor':
        cfg=load(CONFIG);require(bool(cfg.get('api_key')),'API authentication is missing')
        log=Path(cfg['log']);require(log.is_absolute(),'Log path must be absolute')
        # Supports a cursor before the first file is created without opening a nonexistent file.
        if log.exists():mark=mark_log(log)
        else:mark={'time_epoch':time.time(),'boot_id':boot_id(),'uptime_s':uptime(),'log':str(log),'device':None,'inode':None,
                   'offset':0,'anchor_start':0,'anchor_sha256':hashlib.sha256(b'').hexdigest()}
        no_secret(mark,cfg['api_key']);save(a.cursor_file,mark)
        print(json.dumps({'cursor_recorded':True,'cursor_file':str(a.cursor_file),'service_mutations':0}));return
    verify(a.cursor_file,a.out)

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'all_checks_passed':False,'error':str(error) if isinstance(error,RuntimeError) else type(error).__name__}))
        raise SystemExit(1)
