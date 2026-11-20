#!/usr/bin/env python3
"""Read-only Nsight SQLite inventory; schema and numeric aggregates only.

Never prints StringIds, process names, command lines, environment, NVTX/OSRT text,
API arguments or event payloads. Run only after a completed owned export.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

PREFIXES = ('CUPTI_ACTIVITY_KIND_', 'OSRT_API', 'CUDA_GPU_', 'CUDA_GRAPH_', 'PROFILER_', 'COMPOSITE_EVENTS')

def q(name):
    return '"'+name.replace('"','""')+'"'

def inventory(path):
    path=Path(path).resolve()
    if not path.is_file():
        raise ValueError('Existing completed SQLite export required')
    conn=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    conn.execute('PRAGMA query_only=ON')
    try:
        result={'version':'nsys-schema-numeric-1','bytes':path.stat().st_size,
                'read_only':True,'content_strings_exported':False,'tables':[]}
        for name,sql in conn.execute("SELECT name, sql FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
            cols=[{'name':r[1],'declared_type':r[2],'not_null':bool(r[3]),'primary_key_ordinal':r[5]} for r in conn.execute('PRAGMA table_info('+q(name)+')')]
            item={'name':name,'columns':cols,'schema_sql':sql}
            if name.startswith(PREFIXES):
                item['rows']=conn.execute('SELECT count(*) FROM '+q(name)).fetchone()[0]
                names={c['name'] for c in cols}
                for field in ('start','end','timestamp'):
                    if field in names:
                        v=conn.execute('SELECT min('+q(field)+'),max('+q(field)+') FROM '+q(name)).fetchone()
                        if all(x is None or isinstance(x,(int,float)) for x in v):
                            item[field+'_numeric_range']=list(v)
                        else:
                            item[field+'_range_omitted_non_numeric']=True
            result['tables'].append(item)
        return result
    finally:
        conn.close()

def self_test():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'test.sqlite'
        c=sqlite3.connect(p)
        c.execute('CREATE TABLE StringIds (id INTEGER, value TEXT)')
        c.execute('INSERT INTO StringIds VALUES (1,?)',('DO_NOT_PRINT_SECRET',))
        c.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER, globalPid INTEGER)')
        c.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)',[(10,20,0,123),(15,25,1,123)])
        c.commit(); c.close()
        before=hashlib.sha256(p.read_bytes()).hexdigest()
        out=inventory(p)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==before
        assert 'DO_NOT_PRINT_SECRET' not in json.dumps(out)
        event=next(t for t in out['tables'] if t['name']=='CUPTI_ACTIVITY_KIND_KERNEL')
        assert event['rows']==2 and event['start_numeric_range']==[10,15] and event['end_numeric_range']==[20,25]
        assert 'rows' not in next(t for t in out['tables'] if t['name']=='StringIds')
        assert out['read_only']
    return {'self_test':'PASS','checks':['read_only_bytes_unchanged','no_text_values','numeric_ranges','no_general_table_data']}

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('sqlite',nargs='?',type=Path)
    a.add_argument('--out',type=Path)
    a.add_argument('--self-test',action='store_true')
    args=a.parse_args()
    if args.self_test:
        value=self_test()
    else:
        if not args.sqlite: a.error('SQLite path is required')
        value=inventory(args.sqlite)
    text=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if args.out:
        if args.sqlite and args.out.resolve()==args.sqlite.resolve():
            raise ValueError('Output cannot replace input')
        with args.out.open('x') as f: f.write(text)
        args.out.chmod(0o600)
        print(json.dumps({'written':str(args.out),'tables':len(value.get('tables',[]))}))
    else:
        print(text,end='')

if __name__=='__main__': main()
