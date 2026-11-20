#!/bin/bash
# read-only snapshot of the serving box: engine command line / env, GPU clocks and topology, CPU governor, thread affinity (decode latency suspects)
# run:  ssh root@10.100.3.3 bash -s < boxstate.sh
PID=$(pgrep -f '^/srv/Strata-0157/engine/releases/[^ ]*/strata --serve' | head -1)
echo "== engine pid $PID, started $(ps -o lstart= -p $PID)"
echo "== cmdline"; tr '\0' ' ' < /proc/$PID/cmdline | sed -E 's/(--[a-z-]*(key|token|secret)[a-z-]* )[^ ]+/\1<redacted>/g'; echo
echo "== env (STRATA_/CUDA_/malloc/preload)"; tr '\0' '\n' < /proc/$PID/environ | grep -E '^(STRATA_|CUDA_|GLIBC|LD_PRELOAD|OMP_|MALLOC)' | grep -viE 'key|token|secret' | sort | tr '\n' ' '; echo
echo "== uptime / load"; uptime; cat /proc/pressure/cpu 2>/dev/null
echo "== top cpu users"; ps -eo pid,ni,pcpu,nlwp,comm --sort=-pcpu | head -8
echo "== GPUs"
nvidia-smi --query-gpu=index,pstate,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.draw,power.limit,persistence_mode,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,utilization.gpu,temperature.gpu --format=csv,noheader
echo "throttle reasons:"; nvidia-smi --query-gpu=index,clocks_throttle_reasons.active --format=csv,noheader | tr '\n' ' '; echo
echo "== topology"; nvidia-smi topo -m
echo "== p2p read"; nvidia-smi topo -p2p r 2>&1 | head -14
echo "== CPU"; lscpu | grep -E 'Model name|Socket|Thread|Core\(s\)|NUMA|^CPU\(s\)|CPU max|CPU min'
echo "governors:"; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | awk '{s+=$1;n++} END{if(n) printf "avg current MHz %.0f over %d cpus\n", s/n/1000, n}'
echo "pstate driver:"; cat /sys/devices/system/cpu/intel_pstate/status 2>/dev/null; cat /sys/devices/system/cpu/amd_pstate/status 2>/dev/null; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null
echo "idle states:"; ls /sys/devices/system/cpu/cpu0/cpuidle 2>/dev/null | tr '\n' ' '; echo
echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled)  ASPM: $(cat /sys/module/pcie_aspm/parameters/policy 2>/dev/null)"
command -v numactl >/dev/null && numactl -H | head -12
echo "== engine threads (cumulative cpu ticks | last cpu | name | allowed cpus), top 40"
THR=$(for t in /proc/$PID/task/*; do
  n=$(cat $t/comm 2>/dev/null); a=$(awk '/Cpus_allowed_list/{print $2}' $t/status 2>/dev/null)
  s=$(awk -F') ' '{split($2,f," "); print f[12]+f[13]}' $t/stat 2>/dev/null)
  c=$(awk -F') ' '{split($2,f," "); print f[37]}' $t/stat 2>/dev/null)
  echo "$s $c $n $a"
done | sort -rn)
echo "$THR" | head -40
echo "threads in total: $(echo "$THR" | wc -l); allowed-cpu lists:"; echo "$THR" | awk '{print $4}' | sort | uniq -c | sort -rn | head -8
