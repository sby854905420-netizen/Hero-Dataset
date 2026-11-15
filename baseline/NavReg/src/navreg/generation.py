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


class QwenGenerator:
    """vLLM asynchronous engine; qwen3 think boundaries are counted on generated IDs."""
    def __init__(self, config):
        import os
        import importlib.metadata
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
        except ImportError:
            raise RuntimeError('vLLM is not installed; install a checkpoint-compatible GPU build before Qwen trials') from None
        from transformers import AutoTokenizer
        from vllm.config import KernelConfig
        hf, deployment = config['generation']['huggingface'], config['deployment']
        installed = importlib.metadata.version('vllm')
        if hf.get('engine_version') and installed != hf['engine_version']:
            raise ValueError('installed vLLM version differs from the experiment')
        self.profile = hf['sampling_profiles'][hf['profile']]
        self.mode = hf['profile']
        self.max_tokens = deployment['max_new_tokens_by_mode'][self.mode]
        self.input_budget = deployment['input_token_budget']
        self.window = deployment['context_window_tokens']
        self.tokenizer = AutoTokenizer.from_pretrained(deployment['model_id'], revision=deployment['revision'],
                                                       cache_dir=config['paths'].get('model_cache'))
        self.chat_options = {key: self.profile[key] for key in ('enable_thinking', 'preserve_thinking', 'reasoning_effort') if key in self.profile}
        self.start_think = self.tokenizer.encode('<think>', add_special_tokens=False)
        self.end_think = self.tokenizer.encode('</think>', add_special_tokens=False)
        if len(self.start_think) != 1 or len(self.end_think) != 1:
            raise ValueError('qwen3 reasoning requires native single-token think delimiters')
        # Allocate KV from currently free VRAM after embedding and Faiss have loaded.
        import torch
        free, total = torch.cuda.mem_get_info()
        fraction = float(deployment.get('gpu_memory_utilization', 0.85 * free / total))
        args = AsyncEngineArgs(model=deployment['model_id'], revision=deployment['revision'],
                               tokenizer_revision=deployment['revision'], download_dir=config['paths'].get('model_cache'),
                               dtype=deployment['unquantized_dtype'], kv_cache_dtype='auto',
                               max_model_len=self.window, tensor_parallel_size=deployment['tensor_parallel_size'],
                               max_num_seqs=deployment['initial_max_num_seqs'],
                               gpu_memory_utilization=fraction, enforce_eager=deployment.get('enforce_eager', False),
                               generation_config='vllm', language_model_only=True,
                               kernel_config=KernelConfig(**deployment.get('kernel_config', {})),
                               enable_log_requests=False, disable_log_stats=True)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.last_usage = unknown_usage()
        self.identity = {'backend': 'vllm', 'version': installed, 'model': deployment['model_id'],
                         'revision': deployment['revision'], 'sampling': self.profile,
                         'deployment': deployment, 'gpu_memory_utilization': fraction,
                         'reasoning_parser': 'qwen3_generated_token_boundaries', 'language_model_only': True}

    def prepare(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, **self.chat_options)
        if len(ids) > self.input_budget or len(ids) + self.max_tokens > self.window:
            raise ValueError(f'context_overflow: prompt={len(ids)}, input_budget={self.input_budget}, output_reserve={self.max_tokens}, window={self.window}')
        return ids

    async def generate(self, ids, seed, request_id):
        import asyncio
        from vllm import SamplingParams
        options = {key: self.profile[key] for key in ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty')}
        params = SamplingParams(**options, seed=seed, max_tokens=self.max_tokens)
        self.last_usage = unknown_usage(len(ids))
        final = None
        try:
            async for result in self.engine.generate({'prompt_token_ids': ids}, params, request_id):
                final = result
                self.last_usage['observed_output_tokens'] = len(result.outputs[0].token_ids)
        except asyncio.CancelledError:
            await self.engine.abort(request_id)
            raise
        except Exception:
            await self.engine.abort(request_id)
            raise GenerationError('engine_error', usage=self.last_usage) from None
        if final is None:
            raise GenerationError('empty_engine_response', usage=self.last_usage)
        output = final.outputs[0]
        generated = list(output.token_ids)
        usage = {'input_tokens': len(final.prompt_token_ids), 'output_tokens': len(generated),
                 'total_tokens': len(final.prompt_token_ids) + len(generated),
                 'reasoning_tokens': 0, 'cached_input_tokens': getattr(final, 'num_cached_tokens', None)}
        self.last_usage = usage
        if self.end_think[0] in generated:
            boundary = generated.index(self.end_think[0]) + 1
            usage['reasoning_tokens'] = boundary
            visible = generated[boundary:]
        elif self.profile['enable_thinking'] or self.start_think[0] in generated:
            usage['reasoning_tokens'] = None
            raise GenerationError('reasoning_boundary_missing', usage=usage)
        else:
            visible = generated
        text = self.tokenizer.decode(visible, skip_special_tokens=True).strip()
        return {'final_response_text': text, 'usage': usage,
                'status': 'completed' if output.finish_reason == 'stop' else 'failed',
                'metadata': {'model': self.identity['model'], 'finish_reason': output.finish_reason}}

    async def close(self):
        self.engine.shutdown()


def make_generator(config):
    backend = config['generation']['backend']
    if backend == 'openai_api':
        return OpenAIGenerator(config['generation']['openai_api'])
    if backend == 'huggingface':
        return QwenGenerator(config)
    raise ValueError('unsupported generation backend')
