"""The three agreed efficiency averages for one run, including unsuccessful questions."""
import argparse
import json
from pathlib import Path


def summarize(traces, expected_ids):
    rows = {row['question_id']: row for row in traces}
    expected_ids = list(expected_ids)
    n = len(expected_ids)
    sums = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'calls': 0, 'seconds': 0.0}
    coverage = dict.fromkeys(sums, 0)
    for qid in expected_ids:
        row = rows.get(qid)
        if row is None:
            continue
        attempts = row['llm_attempts']
        complete = row.get('attempts_complete', False)
        if complete:
            sums['calls'] += sum(a.get('candidate_count', 1) for a in attempts)
            coverage['calls'] += 1
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            values = []
            for attempt in attempts:
                value = attempt.get(key)
                if key == 'total_tokens':
                    i, o = attempt.get('input_tokens'), attempt.get('output_tokens')
                    value = i + o if i is not None and o is not None else None
                values.append(value)
            sums[key] += sum(v for v in values if v is not None)
            if complete and all(v is not None for v in values):
                coverage[key] += 1
        seconds = row['timing'].get('active_seconds')
        if seconds is not None:
            sums['seconds'] += seconds
            coverage['seconds'] += 1
    def mean(key):
        return sums[key] / n if n and coverage[key] == n else None
    return {'questions': n, 'trace_questions': len(rows),
            'metrics': {'LLM Tokens per Question': {key: mean(key) for key in ('input_tokens', 'output_tokens', 'total_tokens')},
                        'LLM Calls per Question': mean('calls'), 'End-to-end Latency': mean('seconds')},
            'latency_unit': 'seconds', 'known_subtotals': sums, 'complete_question_coverage': coverage,
            'retry_wait_excluded': True, 'judge_excluded': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / 'run_manifest.json').read_text())
    traces = [json.loads(line) for line in (args.run_dir / 'trace.jsonl').read_text().splitlines() if line.strip()]
    result = summarize(traces, manifest['question_order'])
    result['scope'] = manifest['scope']
    (args.run_dir / 'efficiency.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
