"""Score one prediction file; optional judge calls are isolated from the method."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from score import (DEFAULT_DATA, HERE, load_assets, read_jsonl, validate_submissions, load_judgments,
                   evaluate, write_json, write_jsonl)
from summarize import summarize


async def evaluate_predictions(predictions, output_dir, data_dir=DEFAULT_DATA, judgments=None,
                               adjudicate_answers=False, cache_path=None, run_dir=None,
                               policies_path=HERE / 'answer_policies.json', judge_path=HERE / 'judge_config.json'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = None
    if run_dir is not None:
        run_dir = Path(run_dir)
        manifest = json.loads((run_dir / 'run_manifest.json').read_text())
        efficiency = summarize(read_jsonl(run_dir / 'trace.jsonl'), manifest['question_order'])
        efficiency['scope'] = manifest['scope']
        write_json(run_dir / 'efficiency.json', efficiency)
    assets = load_assets(data_dir, policies_path, judge_path)
    submissions, errors = validate_submissions(read_jsonl(predictions), assets)
    if errors:
        write_json(output_dir / 'validation_errors.json', errors)
        raise ValueError('invalid prediction file; see validation_errors.json')
    cache_path = Path(cache_path) if cache_path else output_dir.parent / 'judge_cache.json'
    decisions = load_judgments(judgments, assets['context']) if judgments else {}
    if cache_path.exists():
        decisions.update(load_judgments(cache_path, assets['context']))
    summary, results, pending = evaluate(submissions, assets, decisions)
    if pending and adjudicate_answers and os.environ.get(assets['context']['judge_config']['api_key_env']):
        from judge import adjudicate
        decisions.update(await adjudicate(pending, assets['context'], cache_path, output_dir / 'judge_calls.jsonl'))
        summary, results, pending = evaluate(submissions, assets, decisions)
    if manifest is not None:
        summary['scope'] = manifest['scope']
    write_json(output_dir / 'summary.json', summary)
    write_jsonl(output_dir / 'per_question.jsonl', results)
    write_jsonl(output_dir / 'pending_answer_pairs.jsonl', pending)
    write_json(output_dir / 'run_manifest.json', {**assets['context'], 'predictions': str(Path(predictions).resolve()),
                                               'judge_cache': str(cache_path.resolve())})
    if decisions:
        write_json(output_dir / 'judgments.used.json', {'context': assets['context'], 'decisions': list(decisions.values())})
    return summary


async def evaluate_run(run_dir, adjudicate_answers=True, cache_path=None,
                       policies_path=HERE / 'answer_policies.json', judge_path=HERE / 'judge_config.json'):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'run_manifest.json').read_text())
    if len(manifest['question_order']) < 500:
        efficiency = summarize(read_jsonl(run_dir / 'trace.jsonl'), manifest['question_order'])
        efficiency['scope'] = 'debug'
        write_json(run_dir / 'efficiency.json', efficiency)
        return {'status': 'debug_only', 'questions': len(manifest['question_order']), 'quality': 'full-500 scoring skipped'}
    return await evaluate_predictions(run_dir / 'predictions.jsonl', run_dir / 'metrics',
                                      manifest['config']['paths']['data_root'], adjudicate_answers=adjudicate_answers,
                                      cache_path=cache_path or run_dir.parent / 'judge_cache.json', run_dir=run_dir,
                                      policies_path=policies_path, judge_path=judge_path)


async def main_async():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--judgments', type=Path)
    parser.add_argument('--answer-policies', type=Path, default=HERE / 'answer_policies.json')
    parser.add_argument('--judge-config', type=Path, default=HERE / 'judge_config.json')
    parser.add_argument('--judge-cache', type=Path)
    parser.add_argument('--adjudicate', action='store_true')
    args = parser.parse_args()
    if args.run_dir:
        result = await evaluate_run(args.run_dir, args.adjudicate, args.judge_cache, args.answer_policies, args.judge_config)
    else:
        if not args.predictions or not args.output_dir:
            parser.error('provide --run-dir, or both --predictions and --output-dir')
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            parser.error('output directory is nonempty; use --run-dir to update an existing run or choose a new directory')
        result = await evaluate_predictions(args.predictions, args.output_dir, args.data_dir, args.judgments,
                                            args.adjudicate, args.judge_cache, policies_path=args.answer_policies, judge_path=args.judge_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 3 if result['status'] == 'needs_adjudication' else 0


if __name__ == '__main__':
    try:
        raise SystemExit(asyncio.run(main_async()))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise SystemExit('Cannot evaluate: ' + str(exc))
