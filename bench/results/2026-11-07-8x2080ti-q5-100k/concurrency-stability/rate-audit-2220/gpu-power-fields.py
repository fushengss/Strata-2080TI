#!/usr/bin/env python3
"""Pure local CSV parsing helpers for GPU power/clock-reason audit artifacts."""
import csv
import datetime as dt
import math
from pathlib import Path
import re
import statistics

NUMBER_COLUMNS = {
    'gpu': ('index',),
    'gpu_active_pct': ('utilization.gpu',),
    'memory_active_pct': ('utilization.memory',),
    'memory_used_mib': ('memory.used',),
    'sm_clock_mhz': ('clocks.current.sm', 'clocks.sm'),
    'memory_clock_mhz': ('clocks.current.memory', 'clocks.memory', 'clocks.mem'),
    'graphics_clock_mhz': ('clocks.current.graphics', 'clocks.gr'),
    'power_w': ('power.draw',),
    'power_limit_w': ('power.limit',),
    'enforced_power_limit_w': ('enforced.power.limit',),
    'temperature_c': ('temperature.gpu',),
}
REASONS = ('gpu_idle', 'applications_clocks_setting', 'sw_power_cap', 'hw_slowdown',
           'sync_boost', 'sw_thermal_slowdown', 'hw_thermal_slowdown',
           'hw_power_brake_slowdown', 'display_clock_setting')


def column_name(value):
    return re.sub(r'\s*\[[^]]*\]\s*$', '', value.strip())


def number(value):
    if value is None:
        return None
    match = re.fullmatch(r'\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))(?:\s+[\w%/]+)?\s*', str(value))
    if not match:
        return None
    result = float(match.group(1))
    return result if math.isfinite(result) else None


def active(value):
    if value is None:
        return None
    value = str(value).strip().lower().replace('_', ' ')
    if value in ('active', 'yes', 'true', '1'):
        return True
    if value in ('not active', 'inactive', 'no', 'false', '0'):
        return False
    return None  # Unsupported/missing is not evidence that the reason is inactive.


def read_gpu_csv(path):
    rows, observed_columns = [], set()
    with Path(path).open() as handle:
        for source in csv.DictReader(handle, skipinitialspace=True):
            raw = {column_name(key): value for key, value in source.items()}
            observed_columns.update(raw)
            timestamp = raw.get('timestamp')
            if timestamp:
                epoch = dt.datetime.strptime(timestamp.strip(), '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc).timestamp()
            else:
                epoch = number(raw.get('epoch'))
            if epoch is None:
                raise ValueError('CSV lacks a parseable timestamp/epoch')
            row = {'epoch': epoch}
            for field, aliases in NUMBER_COLUMNS.items():
                row[field] = next((number(raw[key]) for key in aliases if key in raw), None)
            if row['gpu'] is None or not row['gpu'].is_integer():
                raise ValueError('CSV lacks a numeric GPU index')
            row['gpu'] = int(row['gpu'])
            row['clock_reasons'] = {}
            for reason in REASONS:
                aliases = (f'clocks_event_reasons.{reason}', f'clocks_throttle_reasons.{reason}', reason)
                row['clock_reasons'][reason] = next((active(raw[key]) for key in aliases if key in raw), None)
            rows.append(row)
    return rows, sorted(observed_columns)


def summarize_devices(rows, start, end):
    result = []
    for gpu in sorted({row['gpu'] for row in rows}):
        samples = [row for row in rows if row['gpu'] == gpu and start <= row['epoch'] <= end]
        if not samples:
            continue
        entry = {'gpu': gpu, 'n': len(samples), 'first_sample_epoch': samples[0]['epoch'],
                 'last_sample_epoch': samples[-1]['epoch'], 'clock_reasons': {}}
        for field in NUMBER_COLUMNS:
            if field == 'gpu':
                continue
            values = [row[field] for row in samples if row.get(field) is not None]
            entry[field] = {'n': len(values), 'mean': statistics.mean(values), 'min': min(values),
                            'max': max(values)} if values else {'n': 0, 'mean': None, 'min': None, 'max': None}
        for reason in REASONS:
            values = [row['clock_reasons'][reason] for row in samples if row['clock_reasons'][reason] is not None]
            entry['clock_reasons'][reason] = {'valid_n': len(values), 'active_n': sum(values),
                                             'active_pct_of_valid_samples': 100 * sum(values) / len(values) if values else None,
                                             'missing_or_unsupported_n': len(samples) - len(values)}
        result.append(entry)
    return result


if __name__ == '__main__':
    assert active('Active') is True and active('Not Active') is False
    assert active('[Not Supported]') is None and active(None) is None
    assert number('253.27 W') == 253.27 and number('[N/A]') is None
    assert column_name('utilization.gpu [%]') == 'utilization.gpu'
    print('CPU parser checks passed. No files read, network, GPU calls or mutations.')
