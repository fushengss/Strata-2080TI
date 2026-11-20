#!/usr/bin/env python3
"""Record a log cursor before restart, or verify the installed live service. No mutation of service state."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path('/srv/bench100k-20261107-concurrency-mtp')
CONFIG = Path('/srv/Strata/strata-orca-q5_k_m.57.json')
SERVER = Path('/srv/Strata-0157/serve/server.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('cursor', 'verify'))
    a = p.parse_args()
    cfg = json.loads(CONFIG.read_text())
    log = Path(cfg['log'])
    cursor = ROOT / 'deployment-log-cursor.json'
    if a.action == 'cursor':
        require(not cursor.exists(), 'Cursor already exists')
        s = log.stat() if log.exists() else None
        save(cursor, {'time_epoch': time.time(), 'log': str(log), 'device': s.st_dev if s else None,
                      'inode': s.st_ino if s else None, 'offset': s.st_size if s else 0})
        print('Recorded startup log cursor; no service action')
        return
    receipt = json.loads((ROOT / 'deployment-receipt.json').read_text())
    require(all(sha(Path(path)) == digest for path, digest in receipt['installed_sha256'].items()), 'Installed bytes differ')
    request = urllib.request.Request('http://127.0.0.1:8080/v1/status',
                                     headers={'Authorization': 'Bearer ' + cfg['api_key']})
    with urllib.request.urlopen(request, timeout=5) as response:
        status = json.load(response)
    safe_status = {k: status.get(k) for k in ('loaded', 'engine', 'concurrency', 'activity')}
    require(status.get('loaded') is True and status.get('concurrency') == {'serving': 8, 'requested': 8},
            'Production did not load eight serving slots')
    binary = Path(cfg['exe']).resolve()
    library = binary.parent / 'libstrata_kq.so'
    flags = {**receipt['flags'], 'STRATA_DECODE_HEAD_GPU': '2,4', 'STRATA_MTP_HEAD_GPU': '3,5',
             'STRATA_PW_WITH_BATCH': '1', 'STRATA_PREFILL_ZC': '1', 'STRATA_SPLIT_CHUNK_MAX': '8192',
             'STRATA_SPLIT_MTP_BATCH': '1'}
    processes = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc / 'exe').resolve(strict=True) != binary:
                continue
            env = dict(row.split('=', 1) for row in (proc / 'environ').read_bytes().decode().split('\0') if '=' in row)
            maps = (proc / 'maps').read_text()
            require(all(env.get(k) == v for k, v in flags.items()), 'Actual runtime flags differ')
            require(str(library) in maps, 'Matching KQ library is not mapped')
            processes.append({'pid': int(proc.name), 'exe': str(binary), 'engine_sha256': sha(binary),
                              'library_sha256': sha(library), 'flags': {k: env.get(k) for k in flags}})
        except (FileNotFoundError, ProcessLookupError):
            continue
    require(len(processes) == 1, 'Expected one installed engine process')
    mark = json.loads(cursor.read_text())
    require(mark['log'] == str(log), 'Startup log changed')
    stat = log.stat()
    same = (stat.st_dev, stat.st_ino) == (mark['device'], mark['inode']) and stat.st_size >= mark['offset']
    with log.open('rb') as handle:
        if same:
            handle.seek(mark['offset'])
        startup = handle.read()
    parser_path = ROOT / 'harness/runner_head.py'
    spec = importlib.util.spec_from_file_location('frozen_head_runner', parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    class StartupText:
        def read_text(self, errors='strict'):
            return startup.decode(errors=errors)
    # Retain only numerical facts from this restart, never the log text.
    facts = parser.startup_facts(StartupText())
    require(facts['all_experts_resident'] and facts['all_eight_have_3_80_gib_own_buffers']
            and facts['batch_slot_devices'] == list(range(8)) and facts['piped_mtp_startup_enabled'],
            'Fresh startup resource or MTP route gates failed')
    expected = [[8, 3, 5, 7]] if flags['STRATA_PIPED_MTP_HEAD'] == '1' else []
    require(facts['shared_head_workspaces'] == expected, 'Fresh startup shared head route differs')
    require(cfg.get('reasoning_budget_engine_cap') is True, 'Budget cap differs')
    result = {'time_epoch': time.time(), 'all_checks_passed': True, 'status': safe_status,
              'config_sha256': sha(CONFIG), 'server_sha256': sha(SERVER), 'actual_engine': processes,
              'startup_facts': facts, 'fresh_log_offset': mark['offset'] if same else 0,
              'reasoning_budget_engine_cap': True, 'api_key_present': bool(cfg.get('api_key'))}
    save(ROOT / 'deployment-verified.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
