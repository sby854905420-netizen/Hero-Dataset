"""One full experiment, a durable per-attempt journal, and recoverable output views."""
import asyncio
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import time
import uuid

from .config import read_jsonl, write_json, require_path
from .context import SerializedRecords, build_context, parse_answer
from .generation import build_messages, make_generator, GenerationError, unknown_usage
from .retrieval import retrieve


def question_seed(run_seed, question_id, attempt):
    return (run_seed * 1664525 + int(question_id) * 1013904223 + attempt * 69069) % (2 ** 32)


def write_rows(path, rows):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


class Journal:
    def __init__(self, path):
        self.path = path
        self.events = []
        if path.exists():
            # Only a torn final append can be discarded. Earlier corrupt records are errors.
            with path.open('rb+') as stream:
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.endswith(b'\n'):
                        stream.truncate(position)
                        break
                    self.events.append(json.loads(line))

    def append(self, event):
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)

    def materialize(self, directory):
        commits = [event for event in self.events if event['event'] == 'committed']
        write_rows(directory / 'predictions.jsonl', [event['prediction'] for event in commits])
        write_rows(directory / 'trace.jsonl', [event['trace'] for event in commits])


def attempt_record(config, generator, qid, index, run_id):
    seed = config['execution']['run_seed']
    local = config['generation']['backend'] == 'huggingface'
    identity = generator.identity
    return {'request_id': f'{run_id}/{qid}/{index}', 'attempt_id': index,
            'retry_of': index - 1 if index else None, 'run_seed': seed,
            'question_seed': question_seed(seed, qid, index) if local else None,
            'seed_applied': local, 'model_id': identity['model'],
            'model_revision': identity.get('revision'), 'tokenizer_revision': identity.get('revision'),
            'engine': identity.get('version', 'Responses HTTP'),
            'backend': config['generation']['backend'], 'provider': 'local' if local else 'OpenAI',
            'requested_model': identity['model'], 'returned_model': None,
            'task_stage': 'qa', 'purpose': 'answer_generation', 'candidate_count': 1}


async def run_questions(config, questions, directory, journal, documents, embedder, index, generator, run_id):
    committed = {e['question_id'] for e in journal.events if e['event'] == 'committed'}
    policy = config['generation']
    for qid, question in questions:
        if qid in committed:
            continue
        old = [e for e in journal.events if e.get('question_id') == qid]
        ready = next((e for e in reversed(old) if e['event'] == 'ready'), None)
        starts = {e['attempt']['attempt_id']: e for e in old if e['event'] == 'attempt_started'}
        finished = {e['attempt']['attempt_id']: e for e in old if e['event'] == 'attempt_finished'}
        attempts = []
        for number, started in starts.items():
            if number in finished:
                attempts.append(finished[number]['attempt'])
            else:
                attempts.append({**started['attempt'], 'status': 'interrupted_unknown', 'end_perf_s': None,
                                 **unknown_usage(started['attempt'].get('input_tokens')),
                                 'usage_complete': False, 'final_response_text': '', 'retryable': True})
        previous_elapsed = max((e.get('elapsed_seconds', 0) for e in old), default=0)
        previous_wait = sum(e.get('seconds', 0) for e in old if e['event'] == 'retry_wait')
        # A crash inside retrieval or an unclosed request has unobservable elapsed time.
        timing_complete = not old
        start = time.perf_counter()
        wait = 0.0
        retrieval_end = None
        retrieval = ready['retrieval'] if ready else {'retrieved_record_ids': [], 'context_record_ids': [], 'top_chunks': []}
        evidence, labels, error = '', {}, None
        status, answers, citations = 'failed', [], []

        def elapsed():
            return previous_elapsed + time.perf_counter() - start - wait

        if not old:
            journal.append({'event': 'question_started', 'question_id': qid})
        try:
            if not ready:
                retrieval = retrieve(question, embedder, index, config['retrieval']['chunk_k'])
                retrieval_end = time.perf_counter()
                journal.append({'event': 'ready', 'question_id': qid, 'retrieval': retrieval,
                                'elapsed_seconds': elapsed()})
            evidence, labels = build_context(retrieval['context_record_ids'], documents)
            messages = build_messages(question, evidence, prompt_dir=config['prompt_dir'], prompt_files=config.get('prompt_files'))
            prepared = generator.prepare(messages)
            while True:
                last = attempts[-1] if attempts else None
                if last and last['status'] == 'completed':
                    answers, citations = parse_answer(last['final_response_text'], labels)
                    status = 'completed'
                    break
                if last and not last.get('retryable', False):
                    status = last['status'] if last['status'] in ('timeout', 'refused') else 'failed'
                    error = last.get('error')
                    break
                if len(attempts) > policy['retries']:
                    status = 'timeout' if last and last['status'] == 'timeout' else 'failed'
                    error = 'retries_exhausted'
                    break
                if attempts:
                    before = time.perf_counter()
                    await asyncio.sleep(policy['retry_delay_seconds'])
                    seconds = time.perf_counter() - before
                    wait += seconds
                    journal.append({'event': 'retry_wait', 'question_id': qid, 'seconds': seconds,
                                    'elapsed_seconds': elapsed()})
                number = len(attempts)
                attempt = attempt_record(config, generator, qid, number, run_id)
                attempt.update(status='inflight', start_perf_s=time.perf_counter(),
                               **unknown_usage(len(prepared) if isinstance(prepared, list) and (not prepared or isinstance(prepared[0], int)) else None))
                journal.append({'event': 'attempt_started', 'question_id': qid, 'attempt': attempt,
                                'elapsed_seconds': elapsed()})
                generator.last_usage = unknown_usage(attempt['input_tokens'])
                try:
                    response = await asyncio.wait_for(generator.generate(prepared, attempt['question_seed'], attempt['request_id']),
                                                      timeout=policy['request_timeout_seconds'])
                    usage = response['usage']
                    attempt.update(**usage, status=response['status'], final_response_text=response['final_response_text'],
                                   response_metadata=response['metadata'], returned_model=response['metadata'].get('model'),
                                   server_request_id=response['metadata'].get('server_request_id'),
                                   response_id=response['metadata'].get('response_id'), retryable=False)
                except TimeoutError:
                    attempt.update(**generator.last_usage, status='timeout', error='request_timeout', retryable=True,
                                   final_response_text='')
                except GenerationError as exc:
                    attempt.update(**(exc.usage or generator.last_usage), status='failed', error=str(exc),
                                   retryable=exc.retryable, final_response_text='')
                attempt['end_perf_s'] = time.perf_counter()
                attempt['usage_complete'] = attempt['input_tokens'] is not None and attempt['output_tokens'] is not None
                attempt['usage_source'] = 'engine_token_ids' if config['generation']['backend'] == 'huggingface' else 'provider_usage'
                attempt['raw_usage'] = {key: attempt.get(key) for key in ('input_tokens', 'output_tokens', 'total_tokens', 'reasoning_tokens', 'cached_input_tokens', 'observed_output_tokens')}
                journal.append({'event': 'attempt_finished', 'question_id': qid, 'attempt': attempt,
                                'elapsed_seconds': elapsed()})
                attempts.append(attempt)
        except (ValueError, KeyError, TypeError) as exc:
            # Error text here comes from our input/JSON validators, never raw engine output.
            error = 'context_or_response_invalid: ' + str(exc)
            status, answers, citations = 'failed', [], []
        prediction = {'question_id': qid, 'status': status,
                      'retrieved_record_ids': retrieval['retrieved_record_ids'], 'answers': answers, 'citations': citations}
        json.dumps(prediction, ensure_ascii=False)
        end = time.perf_counter()
        trace = {'trace_version': 'navreg-trace-v1', 'question_id': qid, 'status': status,
                 'retrieval': retrieval, 'visible_evidence': evidence, 'citation_labels': labels, 'error': error,
                 'timing': {'question_start_perf_s': start, 'retrieval_end_perf_s': retrieval_end,
                            'question_end_perf_s': end, 'previous_active_seconds': previous_elapsed,
                            'retry_wait_seconds': previous_wait + wait,
                            'active_seconds': previous_elapsed + end - start - wait if timing_complete else None,
                            'known_active_seconds': previous_elapsed + end - start - wait,
                            'timing_complete': timing_complete, 'resumed': bool(old)},
                 'attempts_complete': True, 'llm_attempts': attempts}
        journal.append({'event': 'committed', 'question_id': qid, 'prediction': prediction, 'trace': trace})
        # The journal is authoritative; the two files are recoverable views.
        for name, row in (('predictions.jsonl', prediction), ('trace.jsonl', trace)):
            with (directory / name).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
        print(f'{len(committed) + 1}/{len(questions)} question={qid} status={status}', flush=True)
        committed.add(qid)


async def run_experiment(config, directory=None, index_dir=None, serialized_dir=None, debug=False, limit=None, resume=False):
    from .embedding import Embedder
    from .index import Index, GPU_TEMP_MEMORY_MIB
    if limit is not None and (not debug or limit < 1):
        raise ValueError('--limit requires --debug and a positive sample count')
    data = require_path(config, 'data_root')
    # No answer or annotation object crosses the method boundary.
    questions = {row['question_id']: row['question'] for row in read_jsonl(data / 'qa_pairs.jsonl')}
    if len(questions) != 500:
        raise ValueError('expected 500 QA questions')
    order = sorted(questions)
    random.Random(config['execution']['run_seed']).shuffle(order)
    if limit is not None:
        order = order[:limit]
    bundle = Path(index_dir) if index_dir else require_path(config, 'indexes_root') / config['embedding_name']
    bundle_manifest = json.loads((bundle / 'manifest.json').read_text())
    source = Path(serialized_dir) if serialized_dir else Path(bundle_manifest['serialized_dir'])
    assets = {name: Path(path).read_text() for name, path in config['prompt_files'].items()}
    manifest = {'run_version': 'navreg-run-v1', 'config': config, 'index_dir': str(bundle.resolve()),
                'index_metadata': bundle_manifest, 'serialized_dir': str(source.resolve()), 'prompt_assets': assets,
                'question_order': order, 'questions': [{'question_id': qid, 'question': questions[qid]} for qid in order],
                'scope': 'debug' if debug else 'formal',
                'expected_questions': len(order), 'seed_derivation': 'integer_linear_combination_v1'}
    if directory is None:
        directory = require_path(config, 'runs_root') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    directory = Path(directory)
    if directory.exists() and not resume:
        raise ValueError('run directory exists; use --resume or select a new directory')
    if resume and not (directory / 'run_manifest.json').exists():
        raise ValueError('--resume requires an existing run_manifest.json')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('another process is writing this run') from None
        manifest_path = directory / 'run_manifest.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError('resume requires identical config, prompt, index metadata, seed and question order')
        else:
            write_json(manifest_path, manifest)
        (directory / 'logs').mkdir(exist_ok=True)
        journal = Journal(directory / 'logs/attempts.jsonl')
        journal.materialize(directory)
        done = {e['question_id'] for e in journal.events if e['event'] == 'committed'}
        if len(done) < len(order):
            loaded_at = time.perf_counter()
            documents = SerializedRecords(source / 'records.jsonl')
            embedder = Embedder(config['embedding'], config['paths'].get('model_cache'))
            index = Index(bundle, config)
            generator = make_generator(config)
            try:
                actual_path = directory / 'runtime.json'
                import torch
                actual = {'generator': generator.identity, 'embedding': config['embedding'],
                          'hardware': {'gpu': torch.cuda.get_device_name(0),
                                       'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                                       'torch': torch.__version__, 'cuda': torch.version.cuda},
                          'startup_seconds': time.perf_counter() - loaded_at,
                          'faiss_gpu_temp_memory_mib': GPU_TEMP_MEMORY_MIB}
                if actual_path.exists():
                    old_identity = json.loads(actual_path.read_text())['generator']
                    for key in ('backend', 'model', 'revision', 'version'):
                        if old_identity.get(key) != generator.identity.get(key):
                            raise ValueError('runtime model or engine version changed during resume')
                else:
                    write_json(actual_path, actual)
                print(f'Loaded runtime in {time.perf_counter() - loaded_at:.2f}s; run={directory}', flush=True)
                await run_questions(config, [(qid, questions[qid]) for qid in order], directory, journal, documents,
                                    embedder, index, generator, directory.name)
            finally:
                await generator.close()
        journal.materialize(directory)
    return directory
