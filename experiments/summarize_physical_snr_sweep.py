#!/usr/bin/env python3
import csv, json, math
from pathlib import Path

ROOT = Path('/root/autodl-tmp/btuap-backup-main/results/physical_ota/local_fsc50_bird_snr_sweep_20260921')
RECORDING_ROOT = Path('/root/autodl-tmp/btuap-backup-main/recordings/physical_ota/local_fsc50_bird_snr_sweep_20260921')
RESULTS = ROOT / 'physical_snr_sweep_results.csv'
QC = RECORDING_ROOT / 'recording_qc_flags.csv'
SNRS = [1, 2, 5, 10, 20]

def truth(v):
    return str(v).strip().lower() in {'1','true','yes','y'}

def number(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

def summarize(rows, label):
    out = []
    for snr in SNRS:
        group = [r for r in rows if r['trial_id'].startswith(f'{snr}db_')]
        def unconditional(k):
            return avg([1.0 if truth(r[k]) else 0.0 for r in group])
        def conditional(eligible, success):
            e = [r for r in group if truth(r[eligible])]
            return len(e), sum(truth(r[success]) for r in e), avg([1.0 if truth(r[success]) else 0.0 for r in e])
        esv, ssv, asv = conditional('sv_eligible','sv_success')
        esr, ssr, asr = conditional('sr_eligible','sr_success')
        ej, sj, aj = conditional('joint_eligible','joint_attack_success')
        eu, su, au = conditional('union_eligible','union_attack_success')
        item = {
            'set': label, 'snr_db': snr, 'num_trials': len(group),
            'far': unconditional('sv_accepted'),
            'sensitive_rate': unconditional('is_sensitive'),
            'raw_joint_rate': unconditional('joint_success'),
            'raw_union_rate': unconditional('union_success'),
            'eligible_sv': esv, 'sv_successes': ssv, 'asr_sv': asv,
            'eligible_sr': esr, 'sr_successes': ssr, 'asr_sr': asr,
            'eligible_joint': ej, 'joint_successes': sj, 'asr_joint': aj,
            'eligible_union': eu, 'union_successes': su, 'asr_union': au,
            'mean_sv_clean': avg([number(r.get('sv_score_clean')) for r in group]),
            'mean_sv_ota': avg([number(r.get('sv_score_ota')) for r in group]),
            'mean_sv_delta': avg([(number(r.get('sv_score_ota')) - number(r.get('sv_score_clean'))) if number(r.get('sv_score_ota')) is not None and number(r.get('sv_score_clean')) is not None else None for r in group]),
            'mean_sensitive_clean': avg([number(r.get('p_sensitive_clean')) for r in group]),
            'mean_sensitive_ota': avg([number(r.get('p_sensitive_ota')) for r in group]),
            'mean_sensitive_delta': avg([(number(r.get('p_sensitive_ota')) - number(r.get('p_sensitive_clean'))) if number(r.get('p_sensitive_ota')) is not None and number(r.get('p_sensitive_clean')) is not None else None for r in group]),
        }
        out.append(item)
    return out

with RESULTS.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

qc_ids = set()
if QC.exists():
    with QC.open(newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            snr = str(r.get('snr_db','')).strip().replace('.0','')
            trial = str(r.get('trial_id','')).strip()
            if snr and trial:
                qc_ids.add(f'{snr}db_{trial}')

all_summary = summarize(rows, 'all')
filtered_rows = [r for r in rows if r['trial_id'] not in qc_ids]
filtered_summary = summarize(filtered_rows, 'qc_filtered')
combined = all_summary + filtered_summary
fields = list(combined[0].keys())
with (ROOT/'physical_snr_group_summary.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(combined)
with (ROOT/'physical_snr_group_summary.json').open('w', encoding='utf-8') as f:
    json.dump({'all': all_summary, 'qc_filtered': filtered_summary, 'excluded_trial_ids': sorted(qc_ids)}, f, ensure_ascii=False, indent=2)

def pct(v):
    return 'N/A' if v is None else f'{100*v:.2f}%'
lines = [
    '# Physical OTA FSC50 — JANUS bird SNR sweep',
    '',
    'Frozen thresholds: SV=0.6631416827452248, SR=0.35. Enrollment: Any-10, K=3.',
    '',
    '## All 250 recordings',
    '',
    '| SNR | N | ASR-SV | ASR-SR | ASR-Joint | ASR-Union | FAR | Sensitive Rate |',
    '|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for x in all_summary:
    lines.append(f"| {x['snr_db']} dB | {x['num_trials']} | {pct(x['asr_sv'])} ({x['sv_successes']}/{x['eligible_sv']}) | {pct(x['asr_sr'])} ({x['sr_successes']}/{x['eligible_sr']}) | {pct(x['asr_joint'])} ({x['joint_successes']}/{x['eligible_joint']}) | {pct(x['asr_union'])} ({x['union_successes']}/{x['eligible_union']}) | {pct(x['far'])} | {pct(x['sensitive_rate'])} |")
lines += ['', '## QC sensitivity analysis', '', f'Excluded {len(qc_ids)} pre-identified level outliers: ' + ', '.join(sorted(qc_ids)), '',
          '| SNR | N | ASR-SV | ASR-SR | ASR-Joint | ASR-Union | FAR | Sensitive Rate |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in filtered_summary:
    lines.append(f"| {x['snr_db']} dB | {x['num_trials']} | {pct(x['asr_sv'])} ({x['sv_successes']}/{x['eligible_sv']}) | {pct(x['asr_sr'])} ({x['sr_successes']}/{x['eligible_sr']}) | {pct(x['asr_joint'])} ({x['joint_successes']}/{x['eligible_joint']}) | {pct(x['asr_union'])} ({x['union_successes']}/{x['eligible_union']}) | {pct(x['far'])} | {pct(x['sensitive_rate'])} |")
(ROOT/'physical_snr_group_summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('\n'.join(lines))
