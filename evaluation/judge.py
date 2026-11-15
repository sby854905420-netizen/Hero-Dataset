"""Explicit, separately logged answer adjudication; never part of baseline efficiency."""
import asyncio
import json
import fcntl
import time

from navreg.config import write_json
from navreg.generation import OpenAIGenerator, GenerationError, unknown_usage
from score import canonical


async def adjudicate(pending, context, cache_path, log_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.with_suffix('.lock').open('a') as lock:
        await asyncio.to_thread(fcntl.flock, lock, fcntl.LOCK_EX)
        cache = {'context': context, 'decisions': []}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text())
            if cache['context'] != context:
                raise ValueError('judge cache version differs; use a new cache path')
        decisions = {row['cache_key']: row for row in cache['decisions']}
        config = context['judge_config']
        generator = OpenAIGenerator(config)
        try:
            for pair in pending:
                key = pair['cache_key']
                if key in decisions and decisions[key]['decision'] != 'uncertain':
                    continue
                messages = [{'role': 'system', 'content': context['judge_prompt']},
                            {'role': 'user', 'content': canonical({k: v for k, v in pair.items() if k != 'cache_key'})}]
                for number in range(config['retries'] + 1):
                    if number:
                        before = time.perf_counter()
                        await asyncio.sleep(config['retry_delay_seconds'])
                        with log_path.open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps({'event': 'retry_wait', 'seconds': time.perf_counter() - before}) + '\n')
                    start = time.perf_counter()
                    record = {'cache_key': key, 'attempt_id': number, 'requested_model': config['model_snapshot'],
                              'task_stage': 'evaluation_judge', 'start_perf_s': start, 'status': 'inflight'}
                    with log_path.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                        stream.flush()
                        import os
                        os.fsync(stream.fileno())
                    retryable = False
                    generator.last_usage = unknown_usage()
                    try:
                        response = await asyncio.wait_for(generator.generate(messages, None, f'judge/{pair["question_id"]}/{number}'),
                                                          timeout=config['request_timeout_seconds'])
                        record.update(response)
                        if response['status'] != 'completed':
                            raise ValueError('judge_response_incomplete')
                        answer = json.loads(response['final_response_text'])
                        if (not isinstance(answer, dict) or set(answer) != {'decision', 'reason'} or
                                answer['decision'] not in ('equivalent', 'different', 'uncertain') or
                                not isinstance(answer['reason'], str) or not answer['reason'].strip()):
                            raise ValueError('judge_response_invalid')
                        decisions[key] = {'cache_key': key, **answer}
                        write_json(cache_path, {'context': context, 'decisions': list(decisions.values())})
                    except TimeoutError:
                        record.update(status='timeout', usage=generator.last_usage)
                        retryable = True
                    except GenerationError as exc:
                        record.update(status='failed', error=str(exc), usage=exc.usage or generator.last_usage)
                        retryable = exc.retryable
                    except (ValueError, TypeError, KeyError):
                        record.update(status='failed', error='judge_response_invalid')
                    record['end_perf_s'] = time.perf_counter()
                    with log_path.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    if not retryable:
                        break
        finally:
            await generator.close()
        return decisions
