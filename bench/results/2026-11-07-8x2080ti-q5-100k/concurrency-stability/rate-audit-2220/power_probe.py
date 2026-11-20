import json, pathlib, subprocess, time, hashlib
base=pathlib.Path('/srv/bench100k-20261107-concurrency-mtp')
out=base/'rate-audit-2220'/'c8-power-live'
out.mkdir(exist_ok=False)
cfg_path=pathlib.Path('/srv/Strata/strata-orca-q5_k_m.57.json')
cfg=json.loads(cfg_path.read_text())
before=hashlib.sha256(cfg_path.read_bytes()).hexdigest()
fields='timestamp,index,utilization.gpu,utilization.memory,power.draw,power.limit,enforced.power.limit,clocks.sm,clocks.mem,temperature.gpu,pstate,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown'
start=time.time()
with (out/'gpu.csv').open('w') as f, (out/'gpu-stderr.txt').open('w') as ge, (out/'client-stdout.txt').open('w') as co:
 gpu=subprocess.Popen(['nvidia-smi','--query-gpu='+fields,'--format=csv','-lms','500'],stdout=f,stderr=ge)
 try:
  run=subprocess.run(['python3',str(base/'harness/c8_check.py'),'--execute','--exclusive-benchmark','--config',str(cfg_path),'--port',str(cfg.get('port',8080)),'--expect-piped-mtp','on','--max-tokens','1024','--timeout','120','--out',str(out/'c8')],stdout=co,stderr=subprocess.STDOUT,timeout=160)
  rc=run.returncode
 finally:
  gpu.terminate()
  try: gpu.wait(timeout=5)
  except subprocess.TimeoutExpired: gpu.kill();gpu.wait()
meta={'start_epoch':start,'end_epoch':time.time(),'client_returncode':rc,'config_before_sha256':before,'config_after_sha256':hashlib.sha256(cfg_path.read_bytes()).hexdigest(),'service_restarts':0,'configuration_changes':0,'power_limit_changes':0,'synthetic_requests_requested':8,'tokens_requested_per_request':1024,'gpu_query_fields':fields,'c8_harness_sha256':hashlib.sha256((base/'harness/c8_check.py').read_bytes()).hexdigest()}
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
report=json.loads((out/'c8/c8-check.json').read_text())
print(json.dumps({'client_returncode':rc,'all_checks_passed':report.get('all_checks_passed'),'checks':report.get('checks'),'whole_batch':report.get('timing',{}).get('whole_batch'),'config_unchanged':meta['config_before_sha256']==meta['config_after_sha256']}))

