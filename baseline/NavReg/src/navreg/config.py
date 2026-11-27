"""Resolve one experiment and the shared single-machine runtime configuration."""
from copy import deepcopy
from pathlib import Path
import json
import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'Data'
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
METHOD_FILE = PROJECT_ROOT / 'baseline/NavReg/configs/method.yml'
RUNTIME_FILE = PROJECT_ROOT / 'configs/runtime.yml'


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(runtime=None, experiment=None, seed=None):
    runtime_path = Path(runtime).expanduser().resolve() if runtime else RUNTIME_FILE
    machine = yaml.safe_load(runtime_path.read_text())
    experiment_path = Path(experiment).resolve() if experiment else None
    exp = yaml.safe_load(experiment_path.read_text()) if experiment_path else {}
    method_path = (experiment_path.parent / exp['method_config']).resolve() if exp.get('method_config') else METHOD_FILE
    method = yaml.safe_load(method_path.read_text())
    method = merge(method, {k: exp[k] for k in ('generation', 'retrieval', 'execution') if k in exp})
    if 'chunk_k' in exp:
        method['retrieval']['chunk_k'] = exp['chunk_k']
    paths = {}
    for key, value in machine['paths'].items():
        paths[key] = str((runtime_path.parent / os.path.expandvars(os.path.expanduser(value))).resolve()) if value else None
    method['embedding'] = {**method['embedding_models']['m3'], **{k: method['embedding_models'][k] for k in ('device', 'pooling', 'normalize_embeddings', 'precision', 'batch_size')}}
    if (method['embedding']['device'] != 'cuda' or method['embedding']['precision'] != 'float32'
            or method['embedding']['normalize_embeddings'] is not True):
        raise ValueError('this implementation requires CUDA float32 normalized dense embeddings')
    method['embedding_name'] = 'm3'
    batch = machine.get('execution', {}).get('embedding_batch_size')
    if batch is not None:
        method['embedding']['batch_size'] = batch
    method['execution'] = merge(method['execution'], machine.get('execution', {}))
    if seed is not None:
        method['execution']['run_seed'] = seed
    if type(method['execution']['run_seed']) is not int or method['execution']['run_seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if method['retrieval']['chunk_k'] not in method['retrieval']['chunk_k_values']:
        raise ValueError('unsupported chunk_k')
    if method['execution']['generation_concurrency'] != 1:
        raise ValueError('single-machine execution requires generation_concurrency=1')
    gpu = method['execution'].get('gpu_device')
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    method.update(paths=paths, runtime_config=str(runtime_path))
    method['prompt_dir'] = str(method_path.parent.parent / 'prompts')
    method['prompt_files'] = {key: str((method_path.parent / value).resolve()) for key, value in method['generation']['prompts'].items()}
    budgets = machine.get('generation', {})
    if set(budgets) - {'context_window_tokens', 'input_token_budget', 'max_new_tokens_by_mode'}:
        raise ValueError('runtime generation overrides are limited to context and output budgets')
    method['generation']['huggingface'] = merge(method['generation']['huggingface'], budgets)
    return method


def require_path(config, name):
    value = config['paths'].get(name)
    if not value:
        raise ValueError(f'fill paths.{name} in the environment configuration')
    return Path(value)


def index_settings(config):
    return {'serialization_version': config['serialization']['version'], 'chunking': config['chunking'],
            'embedding': {k: v for k, v in config['embedding'].items() if k != 'batch_size'}, 'index': config['index']}
