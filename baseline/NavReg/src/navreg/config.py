"""Resolve one experiment and one machine profile without separate pipelines."""
from copy import deepcopy
from pathlib import Path
import json
import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'Data'
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
METHOD_FILE = PROJECT_ROOT / 'baseline/NavReg/configs/method.yml'


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


def load_config(profile='local', experiment=None, embedding=None, seed=None):
    profile_path = Path(profile)
    if not profile_path.is_file():
        profile_path = PROJECT_ROOT / 'configs' / (str(profile) + '.yml')
    machine = yaml.safe_load(profile_path.read_text())
    experiment_path = Path(experiment).resolve() if experiment else None
    exp = yaml.safe_load(experiment_path.read_text()) if experiment_path else {}
    method_path = (experiment_path.parent / exp['method_config']).resolve() if exp.get('method_config') else METHOD_FILE
    method = yaml.safe_load(method_path.read_text())
    method = merge(method, {k: exp[k] for k in ('generation', 'retrieval', 'execution') if k in exp})
    if 'chunk_k' in exp:
        method['retrieval']['chunk_k'] = exp['chunk_k']
    choice = embedding or exp.get('embedding', 'large')
    if choice not in ('large', 'm3'):
        raise ValueError('embedding must be large or m3')
    paths = {}
    for key, value in machine['paths'].items():
        paths[key] = str((profile_path.parent / os.path.expandvars(os.path.expanduser(value))).resolve()) if value else None
    method['embedding'] = {**method['embedding_models'][choice], **{k: method['embedding_models'][k] for k in ('device', 'pooling', 'normalize_embeddings', 'precision', 'batch_size')}}
    if (method['embedding']['device'] != 'cuda' or method['embedding']['precision'] != 'float32'
            or method['embedding']['normalize_embeddings'] is not True):
        raise ValueError('this implementation requires CUDA float32 normalized dense embeddings')
    method['embedding_name'] = choice
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
        raise ValueError('first version runs one question at a time; use separate jobs for experiments')
    gpu = method['execution'].get('gpu_device')
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    method.update(paths=paths, profile=machine['profile'])
    method['prompt_dir'] = str(method_path.parent.parent / 'prompts')
    method['prompt_files'] = {key: str((method_path.parent / value).resolve()) for key, value in method['generation']['prompts'].items()}
    hf = method['generation']['huggingface']
    method['deployment'] = hf['deployment_profiles'][hf['deployment_by_environment'][machine['profile']]]
    return method


def require_path(config, name):
    value = config['paths'].get(name)
    if not value:
        raise ValueError(f'fill paths.{name} in the environment configuration')
    return Path(value)


def index_settings(config):
    return {'serialization_version': config['serialization']['version'], 'chunking': config['chunking'],
            'embedding': {k: v for k, v in config['embedding'].items() if k != 'batch_size'}, 'index': config['index']}
