#!/usr/bin/env python3
"""Run the repository's host-only probe under its inherited lease, preserving native records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state(device):
    root = Path('/sys/block') / device
    foreign = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            comm = (proc / 'comm').read_text().strip()
            if comm not in ('strata', 'llama-server', 'llama-cli', 'cc1plus', 'clang-22', 'ninja', 'cmake'):
                continue
            io = dict(line.split(':', 1) for line in (proc / 'io').read_text().splitlines())
            foreign.append(dict(pid=int(proc.name), comm=comm,
                                io={k: int(v) for k, v in io.items()},
                                stat=(proc / 'stat').read_text().strip()))
        except (OSError, ValueError):
            continue
    return dict(time_ns=time.time_ns(), disk_stat=[int(x) for x in (root / 'stat').read_text().split()],
                loadavg=Path('/proc/loadavg').read_text().strip(),
                memory=Path('/proc/meminfo').read_text(), vmstat=Path('/proc/vmstat').read_text(),
                io_pressure=Path('/proc/pressure/io').read_text(),
                cpu_pressure=Path('/proc/pressure/cpu').read_text(), foreign=foreign)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--batches', type=int, default=32)
    p.add_argument('--range-batches', type=int, default=4)
    a = p.parse_args()
    if min(a.rounds, a.batches, a.range_batches) < 1:
        p.error('counts must be positive')
    if not os.environ.get('BLOOMERY_LEASE_HELD'):
        p.error('run through tools/ref/lease-hold.sh')
    sys.path.insert(0, str(a.tree / 'tools'))
    from bloomery import records
    model = Path(os.environ['BLOOMERY_REF_MODEL'])
    source = subprocess.check_output(['findmnt', '-n', '-o', 'SOURCE', '--target', str(model.parent)], text=True).strip()
    parent = subprocess.check_output(['lsblk', '-n', '-o', 'PKNAME', source], text=True).strip()
    device = parent or Path(source).name
    a.out.mkdir(parents=True, exist_ok=True)
    disk = Path('/sys/block') / device
    metadata = dict(tree=str(a.tree), commit=subprocess.check_output(['git', '-C', str(a.tree), 'rev-parse', 'HEAD'], text=True).strip(),
                    binary=str(a.binary), binary_sha256=digest(a.binary), model=str(model),
                    model_bytes=model.stat().st_size, device=device,
                    drive_model=(disk / 'device/model').read_text().strip(), attachment='NVMe; model filesystem ext4',
                    queue={name: (disk / 'queue' / name).read_text().strip() for name in ('scheduler', 'read_ahead_kb', 'max_sectors_kb')},
                    rounds=a.rounds, batches=a.batches, warmup=2, ks=[1, 2, 4, 8], threads=[1, 2, 4, 8],
                    range_from=480, range_batches=a.range_batches, direct_only=True,
                    scope='Native probe O_DIRECT file IO only. Workers are an upper bound on outstanding blocking requests, not measured hardware queue depth. One directN read per batch; range drive reference follows directN on the same bytes. SSD/controller caches not reset. Foreign load is recorded, not stopped.',
                    lease_directory=os.environ.get('BLOOMERY_LOCK_DIR'), kernel=os.uname().release)
    metadata['probe_source_sha256'] = digest(a.tree / 'crates/gpu-gates/src/bin/probe_nvread.rs')
    metadata['probe_diff_from_main'] = subprocess.check_output(
        ['git', '-C', str(a.tree), 'diff', 'origin/main', '--', 'crates/gpu-gates/src/bin/probe_nvread.rs'], text=True)
    metadata['working_tree_status'] = subprocess.check_output(['git', '-C', str(a.tree), 'status', '--short'], text=True)
    metadata['sweep_script_sha256'] = digest(__file__)
    (a.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    results = []
    for rnd in range(a.rounds):
        threads = [1, 2, 4, 8]
        threads = threads[rnd % 4:] + threads[:rnd % 4]
        seed = 11071 + rnd
        for count in threads:
            stem = f'r{rnd + 1}-t{count}'
            cmd = [str(a.binary), '--ks', '1,2,4,8', '--threads', str(count), '--batches', str(a.batches),
                   '--warmup', '2', '--range-from', '480', '--range-batches', str(a.range_batches),
                   '--seed', str(seed), '--lease', '--direct-only']
            before = state(device)
            with (a.out / (stem + '.log')).open('w') as f:
                f.write('command: ' + ' '.join(cmd) + '\n'); f.flush()
                start = time.monotonic_ns()
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=180)
                elapsed = time.monotonic_ns() - start
            after = state(device)
            recs = records.read(a.out / (stem + '.log'), bin_name='probe_nvread')
            rows = [dict(rec.fields) for rec in recs if rec.kind == 'nvread']
            if any(rec.loose for rec in recs):
                raise RuntimeError(f'{stem}: record schema did not read whole')
            asked = sum(row['read_bytes'] for row in rows)
            physical = (after['disk_stat'][2] - before['disk_stat'][2]) * 512
            ratio = physical / asked if asked else None
            good = r.returncode == 0 and len(rows) == 6 and ratio is not None and .95 <= ratio <= 1.05
            result = dict(round=rnd + 1, threads=count, seed=seed, command=cmd, rc=r.returncode,
                          elapsed_ns=elapsed, before=before, after=after, rows=rows,
                          probe_read_bytes=asked, device_read_bytes=physical, reconcile_ratio=ratio,
                          reconciled=good)
            results.append(result)
            (a.out / (stem + '.json')).write_text(json.dumps(result, indent=2) + '\n')
            (a.out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(f'{stem}: rc={r.returncode} rows={len(rows)} device/probe={ratio} reconciled={good}', flush=True)
            if r.returncode:
                print((a.out / (stem + '.log')).read_text()[-5000:], flush=True)
                return r.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
