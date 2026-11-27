"""Fixed messages and inference adapters; never persist generated thinking."""
import json
from pathlib import Path
import re


def build_messages(
    question: str,
    evidence: str,
    *,
    prompt_dir: str | Path | None = None,
    prompt_files: dict | None = None,
) -> list[dict[str, str]]:
    """Build the same fixed three-shot messages for local and API generation.

    The default templates belong to the source checkout. An installation that
    stores templates elsewhere must pass its template directory explicitly.
    Only the current question, retrieved evidence and three public examples
    enter messages. Example provenance stays local to the asset; this function
    never reads QA pairs, standard answers or evaluation annotations.
    """
    template_dir = (
        Path(prompt_dir)
        if prompt_dir is not None
        else Path(__file__).resolve().parents[2] / "prompts"
    )
    files = prompt_files or {"system_file": template_dir / "system.txt", "user_file": template_dir / "answer.txt", "examples_file": template_dir / "examples.json"}
    system = Path(files["system_file"]).read_text(encoding="utf-8").strip()
    user_template = Path(files["user_file"]).read_text(encoding="utf-8")
    if not system or any(
        user_template.count("{{" + name + "}}") != 1
        for name in ("question", "evidence")
    ):
        raise ValueError("Expected a nonempty system prompt and one placeholder per input")

    def render_user(example_question: str, example_evidence: str) -> str:
        values = {"question": example_question, "evidence": example_evidence}
        # Substitute once so placeholder-like source text remains literal data.
        return re.sub(
            r"\{\{(question|evidence)\}\}",
            lambda match: values[match.group(1)],
            user_template,
        )

    asset = json.loads(Path(files["examples_file"]).read_text(encoding="utf-8"))
    if (
        not isinstance(asset, dict)
        or asset.get("version") != "navreg-few-shot-v1"
        or not isinstance(asset.get("examples"), list)
        or len(asset["examples"]) != 3
    ):
        raise ValueError("Expected navreg-few-shot-v1 with exactly three examples")

    messages = [{"role": "system", "content": system}]
    example_ids: set[str] = set()
    for example in asset["examples"]:
        if not isinstance(example, dict) or any(
            not isinstance(example.get(name), str) or not example[name].strip()
            for name in ("example_id", "question", "evidence")
        ) or not isinstance(example.get("provenance"), dict):
            raise ValueError("Each example needs an ID, question, evidence and provenance")
        if example["example_id"] in example_ids:
            raise ValueError("Example IDs must be unique")
        example_ids.add(example["example_id"])

        response = example.get("response")
        if not isinstance(response, dict) or set(response) != {"answers", "citations"}:
            raise ValueError("Example responses must contain only answers and citations")
        for name in ("answers", "citations"):
            items = response[name]
            if (
                not isinstance(items, list)
                or not items
                or any(not isinstance(item, str) or not item.strip() for item in items)
                or len(set(items)) != len(items)
            ):
                raise ValueError("Example answers and citations must be nonempty unique strings")
        declarations = re.findall(
            r"^\[([A-Za-z][A-Za-z0-9_]*)\] Citable: (true|false)$",
            example["evidence"],
            flags=re.MULTILINE,
        )
        labels = [label for label, _ in declarations]
        citable = {label for label, visible in declarations if visible == "true"}
        if (
            not labels
            or len(set(labels)) != len(labels)
            or any(not label.startswith("DEMO") for label in labels)
            or not set(response["citations"]).issubset(citable)
        ):
            raise ValueError("Examples must cite distinct declared citable DEMO labels")
        messages.extend(
            [
                {"role": "user", "content": render_user(example["question"], example["evidence"])},
                {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
            ]
        )
    messages.append({"role": "user", "content": render_user(question, evidence)})
    return messages


class GenerationError(Exception):
    def __init__(self, code, retryable=False, usage=None):
        super().__init__(code)
        self.retryable = retryable
        self.usage = usage


def unknown_usage(input_tokens=None):
    return {'input_tokens': input_tokens, 'output_tokens': None, 'total_tokens': None,
            'reasoning_tokens': None, 'cached_input_tokens': None}


def _qwen_response(generator, generated, usage, finish_reason):
    if generator.end_think[0] in generated:
        boundary = generated.index(generator.end_think[0]) + 1
        usage['reasoning_tokens'] = boundary
        visible = generated[boundary:]
    elif generator.profile['enable_thinking'] or generator.start_think[0] in generated:
        usage['reasoning_tokens'] = None
        raise GenerationError('reasoning_boundary_missing', usage=usage)
    else:
        usage['reasoning_tokens'] = 0
        visible = generated
    return {'final_response_text': generator.tokenizer.decode(visible, skip_special_tokens=True).strip(),
            'usage': usage, 'status': 'completed' if finish_reason == 'stop' else 'failed',
            'metadata': {'model': generator.identity['model'], 'finish_reason': finish_reason}}


class OpenAIGenerator:
    """Responses HTTP transport with no hidden retries, tools or conversation state."""
    def __init__(self, config):
        import os
        import httpx
        self.config = config
        key = os.environ.get(config.get('api_key_env', 'OPENAI_API_KEY'))
        if not key:
            raise ValueError('set ' + config.get('api_key_env', 'OPENAI_API_KEY') + ' before using the API backend')
        self.client = httpx.AsyncClient(base_url=config.get('base_url', 'https://api.openai.com/v1').rstrip('/') + '/',
                                        headers={'Authorization': 'Bearer ' + key}, timeout=None)
        self.last_usage = unknown_usage()
        self.identity = {'backend': 'openai_api', 'model': config.get('model_id', config.get('model_snapshot')),
                         'parameters': config.get('parameters', {}), 'budget_check': 'provider_context_limit_no_local_token_estimate'}

    def prepare(self, messages):
        return messages

    async def generate(self, messages, seed, request_id):
        import httpx
        self.last_usage = unknown_usage()
        payload = {'model': self.identity['model'], 'input': messages, **self.config.get('parameters', {}),
                   'tools': [], 'tool_choice': 'none', 'store': False}
        try:
            response = await self.client.post('responses', json=payload)
        except httpx.TransportError:
            raise GenerationError('transport_error', retryable=True) from None
        if response.status_code >= 400:
            raise GenerationError('http_' + str(response.status_code),
                                  retryable=response.status_code in (408, 429, 500, 502, 503, 504))
        try:
            body = response.json()
            raw = body.get('usage') or {}
            usage = {'input_tokens': raw.get('input_tokens'), 'output_tokens': raw.get('output_tokens'),
                     'total_tokens': raw.get('total_tokens'),
                     'reasoning_tokens': (raw.get('output_tokens_details') or {}).get('reasoning_tokens'),
                     'cached_input_tokens': (raw.get('input_tokens_details') or {}).get('cached_tokens')}
            self.last_usage = usage
            parts, refusal = [], False
            for item in body.get('output', []):
                if item.get('type') != 'message':
                    continue
                for content in item.get('content', []):
                    if content.get('type') == 'output_text':
                        parts.append(content['text'])
                    elif content.get('type') == 'refusal':
                        refusal = True
            return {'final_response_text': ''.join(parts), 'usage': usage,
                    'status': 'refused' if refusal else ('completed' if body.get('status') == 'completed' else 'failed'),
                    'metadata': {'response_id': body.get('id'), 'model': body.get('model'),
                                 'provider_status': body.get('status'),
                                 'server_request_id': response.headers.get('x-request-id'),
                                 'temperature': body.get('temperature'), 'top_p': body.get('top_p'),
                                 'max_output_tokens': body.get('max_output_tokens'),
                                 'reasoning_effort': (body.get('reasoning') or {}).get('effort')}}
        except (ValueError, KeyError, TypeError, AttributeError):
            raise GenerationError('invalid_api_response', usage=self.last_usage) from None

    async def close(self):
        await self.client.aclose()


def _fla_identity():
    import importlib.metadata
    import inspect

    try:
        versions = {name: importlib.metadata.version(name) for name in ('fla-core', 'triton')}
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    except ImportError:
        raise ValueError('FLA acceleration requires fla-core and Triton; install the baseline dependencies in the active GPU environment') from None
    from transformers.models.qwen3_5 import modeling_qwen3_5

    kernels = {}
    for name, wrapped, implementation in (
        ('chunk', modeling_qwen3_5.torch_chunk_gated_delta_rule, chunk_gated_delta_rule),
        ('recurrent', modeling_qwen3_5.torch_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule),
    ):
        # Transformers preserves the reference function's name even when FLA is bound.
        while inspect.getclosurevars(wrapped).nonlocals.get('implementation') is not implementation:
            wrapped = getattr(wrapped, '__wrapped__', None)
            if wrapped is None:
                raise ValueError(f'FLA {name} kernel is not active; repair the GPU environment and start a new Python process')
        kernels[name] = implementation.__module__ + '.' + implementation.__name__
    return {'backend': 'fla-core', 'version': versions['fla-core'],
            'triton_version': versions['triton'], 'kernels': kernels}


class TransformerGenerator:
    """Single-GPU prequantized NF4 inference; cancellation drains the worker."""
    def __init__(self, config):
        import asyncio
        import importlib.metadata
        import torch
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration

        hf = config['generation']['huggingface']
        installed = importlib.metadata.version('transformers')
        expected = hf.get('engine_version')
        if expected and installed.split('+', 1)[0] != expected:
            raise ValueError('installed Transformers version differs from the experiment')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise ValueError('Transformers BF16 generation requires one visible BF16-capable CUDA GPU')
        linear_attention = _fla_identity()
        if hf['unquantized_dtype'] != 'bfloat16' or hf['quantization'] != 'bitsandbytes_nf4':
            raise ValueError('Transformers requires a prequantized NF4 checkpoint with BF16 compute')
        self.profile = hf['sampling_profiles'][hf['profile']]
        self.mode = hf['profile']
        self.max_tokens = hf['max_new_tokens_by_mode'][self.mode]
        self.input_budget = hf['input_token_budget']
        self.window = hf['context_window_tokens']
        output_budgets = list(hf['max_new_tokens_by_mode'].values())
        if (any(type(value) is not int or value < 1 for value in [self.window, self.input_budget, *output_budgets])
                or self.window > 262144 or self.input_budget + max(output_budgets) > self.window):
            raise ValueError('context budgets must be positive integers with input plus output within the window and native 262144-token limit')
        self.timeout = config['generation']['request_timeout_seconds']
        self._lock = asyncio.Lock()
        loading = {'revision': hf['revision'], 'cache_dir': config['paths'].get('model_cache')}
        checkpoint = AutoConfig.from_pretrained(hf['model_id'], **loading)
        quantization = getattr(checkpoint, 'quantization_config', None)
        from bitsandbytes.nn import Linear4bit
        bnb_version = importlib.metadata.version('bitsandbytes')
        if bnb_version != hf['bitsandbytes_version']:
            raise ValueError('installed bitsandbytes version differs from the experiment')
        if (not isinstance(quantization, dict) or quantization.get('quant_method') != 'bitsandbytes'
                or not quantization.get('load_in_4bit') or quantization.get('bnb_4bit_quant_type') != 'nf4'
                or quantization.get('bnb_4bit_use_double_quant') is not True):
            raise ValueError('expected a prequantized bitsandbytes NF4 double-quant checkpoint')
        # Keep the publisher's skipped layers; override only arithmetic precision.
        checkpoint.quantization_config = {**quantization, 'bnb_4bit_compute_dtype': 'bfloat16'}
        self.tokenizer = AutoTokenizer.from_pretrained(hf['model_id'], **loading)
        self.chat_options = {key: self.profile[key] for key in ('enable_thinking', 'preserve_thinking', 'reasoning_effort') if key in self.profile}
        self.start_think = self.tokenizer.encode('<think>', add_special_tokens=False)
        self.end_think = self.tokenizer.encode('</think>', add_special_tokens=False)
        if len(self.start_think) != 1 or len(self.end_think) != 1:
            raise ValueError('qwen3 reasoning requires native single-token think delimiters')
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            hf['model_id'], config=checkpoint, dtype=torch.bfloat16,
            device_map={'': 0}, attn_implementation='sdpa', use_kernels=False, **loading,
        ).eval()
        for parameter in self.model.parameters():
            if parameter.device.type != 'cuda' or parameter.device.index != 0:
                raise ValueError('all generation parameters must reside on CUDA device 0; offload is not supported')
            if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
                raise ValueError('all floating generation parameters must be BF16')
        modules = [module for module in self.model.modules() if isinstance(module, Linear4bit)]
        if not getattr(self.model, 'is_loaded_in_4bit', False) or not modules:
            raise ValueError('the requested NF4 checkpoint was not loaded as four-bit Linear layers')
        if any(module.compute_dtype != torch.bfloat16 or module.weight.quant_type != 'nf4'
               or module.weight.quant_state is None or not module.weight.quant_state.nested for module in modules):
            raise ValueError('actual NF4 arithmetic or nested quantization differs from the configured NF4 compute')
        actual_quantization = self.model.config.quantization_config
        quantization_identity = dict(method='bitsandbytes_nf4', version=bnb_version,
                                     compute_dtype='bfloat16', linear4bit_modules=len(modules),
                                     config=actual_quantization.to_dict() if hasattr(actual_quantization, 'to_dict') else actual_quantization)
        self.last_usage = unknown_usage()
        self.identity = {'backend': 'transformers', 'version': installed, 'model': hf['model_id'],
                         'revision': hf['revision'], 'sampling': self.profile,
                         'context_window_tokens': self.window, 'input_token_budget': self.input_budget,
                         'max_new_tokens': self.max_tokens,
                         'dtype': 'bfloat16', 'device': 'cuda:0', 'attn_implementation': 'sdpa',
                         'use_kernels': False, 'disable_compile': True, 'quantization': quantization_identity,
                         'linear_attention': linear_attention,
                         'reasoning_parser': 'qwen3_generated_token_boundaries',
                         'presence_penalty_scope': 'generated_tokens_only', 'input_modality': 'text'}

    def prepare(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, **self.chat_options)
        if len(ids) > self.input_budget or len(ids) + self.max_tokens > self.window:
            raise ValueError(f'context_overflow: prompt={len(ids)}, input_budget={self.input_budget}, output_reserve={self.max_tokens}, window={self.window}')
        return ids

    def _generate(self, ids, seed, stop, expired, deadline):
        import time
        import torch
        from transformers import LogitsProcessor, StoppingCriteria

        owner = self

        class PresencePenalty(LogitsProcessor):
            def __call__(self, input_ids, scores):
                # Presence counts each generated token once and excludes the input prompt.
                seen = input_ids[0, len(ids):].unique()
                scores[0, seen] -= owner.profile['presence_penalty']
                return scores

        class StopGeneration(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                owner.last_usage['observed_output_tokens'] = input_ids.shape[-1] - len(ids)
                if time.monotonic() >= deadline:
                    expired.set()
                return stop.is_set() or expired.is_set()

        options = {key: self.profile[key] for key in ('do_sample', 'temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty')}
        processors = [PresencePenalty()] if self.profile['presence_penalty'] else []
        try:
            with torch.inference_mode():
                torch.manual_seed(seed)
                inputs = torch.tensor([ids], dtype=torch.long, device='cuda:0')
                output = self.model.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    max_new_tokens=self.max_tokens, num_beams=1, num_return_sequences=1,
                    logits_processor=processors, stopping_criteria=[StopGeneration()],
                    use_cache=True, cache_implementation='dynamic',
                    disable_compile=True, return_dict_in_generate=False, **options,
                )
                generated = output[0, len(ids):].tolist()
                self.last_usage = {'input_tokens': len(ids), 'output_tokens': len(generated),
                                   'total_tokens': len(ids) + len(generated),
                                   'reasoning_tokens': None, 'cached_input_tokens': 0}
                return generated
        finally:
            # A cancelled asyncio task must not release this GPU to the next attempt yet.
            torch.cuda.synchronize(0)

    async def generate(self, ids, seed, request_id):
        import asyncio
        import threading
        import time

        async with self._lock:
            self.last_usage = unknown_usage(len(ids))
            stop, expired = threading.Event(), threading.Event()
            worker = asyncio.create_task(asyncio.to_thread(
                self._generate, ids, seed, stop, expired, time.monotonic() + self.timeout,
            ))
            try:
                generated = await asyncio.shield(worker)
            except asyncio.CancelledError:
                stop.set()
                try:
                    await asyncio.shield(worker)
                except Exception:
                    pass
                raise
            except Exception:
                raise GenerationError('engine_error', usage=self.last_usage) from None
            if expired.is_set():
                raise TimeoutError('request_timeout')
            if not generated:
                raise GenerationError('empty_engine_response', usage=self.last_usage)
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, (list, tuple)) else [eos]
            finish_reason = 'stop' if generated[-1] in eos else 'length'
            return _qwen_response(self, generated, self.last_usage, finish_reason)

    async def close(self):
        import torch
        async with self._lock:
            self.model = None
            torch.cuda.empty_cache()


def make_generator(config):
    backend = config['generation']['backend']
    if backend == 'openai_api':
        return OpenAIGenerator(config['generation']['openai_api'])
    if backend == 'huggingface':
        return TransformerGenerator(config)
    raise ValueError('unsupported generation backend')
