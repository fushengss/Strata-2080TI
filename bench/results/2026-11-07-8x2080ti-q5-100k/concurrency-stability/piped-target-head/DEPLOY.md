# Piped decode installer

`deploy_piped_decode.py` is prepared locally only. No remote deployment or service operation was performed by this preparation task. The final binary and switches must be selected from completed correctness, concurrency and 100k prefill validation.

On the test host, supply all four choices explicitly:

```sh
python3 deploy_piped_decode.py \
  --candidate /absolute/path/to/validated/strata.CANDIDATE \
  --engine-sha256 VALIDATED_64_CHARACTER_SHA256 \
  --target-head 1 --async-sample 0
```

Without an action flag this only prints a plan, performs no host checks and writes nothing. Add `--check` to validate files/configuration and maintenance state without writes. After validation and selection, add `--execute` to install. `0` or `1` is required separately for each switch; neither option is inferred from a filename.

The host must already have no GPU compute processes and no listener on port 8080 or 8092, including non-loopback/IPv6 listeners. The script does not stop or start anything. It leaves the existing GPU clock lock and power limits untouched.

Frozen prerequisites: original config `f5376cc8…`, unchanged server `40c5e9df…`, adjacent `libstrata_kq.so` `fb67bffc…`; complete hashes are constants in the script. Only `exe`, `env.STRATA_PIPED_TARGET_HEAD`, and `env.STRATA_PIPED_ASYNC_SAMPLE` may change. Every other configuration/environment field, including the API key, layer split, model and prefill controls, is compared for equality. Server bytes are never rewritten.

Installation destination is uniquely `/srv/Strata-0157/engine/releases/piped-decode-20261107`; an existing destination is rejected. A private `rollback-piped-decode-*` directory under `/srv/bench100k-20261107-piped-target-head` stores original config/server bytes (directory 0700, files 0600). The active config is atomically replaced with mode 0600 only after release verification and repeated quiescence/original-file checks. A caught post-write failure, SIGINT or SIGTERM restores the original config bytes atomically. SIGKILL/power loss cannot run rollback; the durable private backup remains available. An unreferenced failed release may remain and is never overwritten automatically.

Success writes a key-free `/srv/bench100k-20261107-piped-target-head/deployment-receipt.json` containing paths, hashes, selected flags and preservation checks. It records that services have not been started. Inspect and start the existing service through the established maintenance procedure separately; do not use a restore action that overwrites this newly installed config.

Local check: `python3 deploy_piped_decode.py --self-test`. Temporary fake files test protected-field rejection, SHA mismatch, a post-config injected failure and restoration, duplicate release refusal, successful install, private modes, and receipt secrecy. Result: `deploy-offline-check.json`. `deploy-dry-run-example.json` uses a nonexistent fake candidate to demonstrate default mode has no host/file dependency; it is not a deployment choice.
