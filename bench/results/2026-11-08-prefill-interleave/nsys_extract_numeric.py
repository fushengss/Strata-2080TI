#!/usr/bin/env python3
"""Stream only numeric activity rows plus validated CUDA API identifiers from owned export."""
import gzip,json,re,sqlite3,struct,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve(); pid=int(sys.argv[2]); c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON')
processes=list(c.execute('SELECT globalPid,pid FROM PROCESSES'))
gp=[r[0] for r in processes if r[1]==pid]
assert len(gp)==1, 'Unique owned engine process required'
metadata={'version':'nsys-numeric-stream-1','engine_pid':pid,'engine_global_pid':gp[0],'processes_numeric':processes,
 'devices_numeric':list(c.execute('SELECT gpuId,cudaId,pid FROM TARGET_INFO_CUDA_DEVICE')),
 'gpu_numeric':list(c.execute('SELECT id,cuDevice,smCount,totalMemory FROM TARGET_INFO_GPU')),
 'session_epoch_ns':list(c.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME')),
 'analysis_numeric':list(c.execute('SELECT globalVid,duration,startTime,stopTime FROM ANALYSIS_DETAILS')),
 'diagnostics_numeric':list(c.execute('SELECT timestamp,timestampType,source,severity,globalPid, CASE WHEN lower(text) LIKE \'%lost%\' OR lower(text) LIKE \'%dropped%\' THEN 1 ELSE 0 END FROM DIAGNOSTIC_EVENT')),
 'record_format':'little endian8signed64:kind,start,end,device_or_minus1,globalPid_or_globalTid,correlation_or_nameId,graphNode_or_nameId_or_syncType,shortName_or_bytes_or_returnValue_or_eventId',
 'kind':{'1':'kernel','2':'memcpy','3':'memset','4':'runtime','5':'synchronization','6':'osrt','7':'profiler'}}
metadata['cuda_api_names']={str(i):v for i,v in c.execute('SELECT DISTINCT s.id,s.value FROM StringIds s JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.nameId=s.id') if re.fullmatch(r'cuda[A-Za-z0-9_]+',v)}
fields=[(1,'KERNEL','start,end,deviceId,globalPid,correlationId,graphNodeId,shortName'),(2,'MEMCPY','start,end,deviceId,globalPid,correlationId,graphNodeId,bytes'),(3,'MEMSET','start,end,deviceId,globalPid,correlationId,graphNodeId,bytes'),(4,'RUNTIME','start,end,-1,globalTid,correlationId,nameId,returnValue'),(5,'SYNCHRONIZATION','start,end,deviceId,globalPid,correlationId,syncType,eventId'),(6,'OSRT_API','start,end,-1,globalTid,nameId,returnValue,nestingLevel'),(7,'PROFILER_OVERHEAD','start,end,-1,globalTid,nameId,returnValue,-1')]
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1,mtime=0) as out:
 out.write(b'NSYSNUM1\n'+json.dumps(metadata,separators=(',',':')).encode()+b'\n')
 pack=struct.Struct('<8q').pack
 for kind,table,cols in fields:
  table=table if kind>=6 else 'CUPTI_ACTIVITY_KIND_'+table
  where=' WHERE globalPid=?' if kind in (1,2,3,5) else ''
  cur=c.execute('SELECT '+cols+' FROM '+table+where, (gp[0],) if where else ())
  while True:
   rows=cur.fetchmany(16384)
   if not rows: break
   out.write(b''.join(pack(kind,*(x if x is not None else -1 for x in row)) for row in rows))
c.close()
