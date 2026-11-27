"""One immutable exact Faiss bundle per embedding model; one GPU owner per job."""
import json
from pathlib import Path
import tempfile
import shutil

from .config import read_jsonl, write_json, index_settings, require_path
from .chunking import chunk_documents

GPU_TEMP_MEMORY_MIB = 64


def build_index(config, output_dir=None, serialized_dir=None):
    import faiss
    from .embedding import Embedder
    from .prepare import serialize_repository
    target = Path(output_dir) if output_dir else require_path(config, 'indexes_root') / config['embedding_name']
    if target.exists():
        raise ValueError(f'index directory already exists; use a new name: {target}')
    source = Path(serialized_dir) if serialized_dir else require_path(config, 'serialized_root') / config['serialization']['version']
    if not source.exists():
        serialize_repository(require_path(config, 'data_root'), source)
    encoder = Embedder(config['embedding'], config['paths'].get('model_cache'))
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=target.name + '.building-', dir=target.parent))
    try:
        resources = faiss.StandardGpuResources()
        resources.setTempMemory(GPU_TEMP_MEMORY_MIB * 1024 ** 2)
        cpu = faiss.IndexFlatIP(config['embedding']['dimensions'])
        gpu = faiss.index_cpu_to_gpu(resources, 0, cpu)
        count, parent_count = 0, 0
        batch = []
        with (source / 'records.jsonl').open(encoding='utf-8') as records, (stage / 'chunk_map.jsonl').open('w', encoding='utf-8') as mapping:
            def documents():
                nonlocal parent_count
                for line in records:
                    if line.strip():
                        parent_count += 1
                        yield json.loads(line)

            def add():
                nonlocal count
                gpu.add(encoder.encode([item['text'] for item in batch]))
                for item in batch:
                    mapping.write(json.dumps({'vector_index': count, **item}, ensure_ascii=False) + '\n')
                    count += 1
                batch.clear()
            for item in chunk_documents(documents(), encoder.tokenizer, config['embedding']['max_input_tokens'], config['chunking']['cell_value_max_tokens']):
                batch.append(item)
                if len(batch) >= encoder.batch_size:
                    add()
            if batch:
                add()
        if not count:
            raise ValueError('no indexable chunks')
        faiss.write_index(faiss.index_gpu_to_cpu(gpu), str(stage / 'index.faiss'))
        manifest = {'bundle_version': 'navreg-index-v1', 'settings': index_settings(config),
                    'vector_count': count, 'parent_count': parent_count,
                    'serialized_dir': str(source.resolve()), 'data_root': str(require_path(config, 'data_root'))}
        write_json(stage / 'manifest.json', manifest)
        stage.rename(target)
        return {'index_dir': str(target), 'vector_count': count, 'parent_count': parent_count}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


class Index:
    def __init__(self, directory, config):
        import faiss
        directory = Path(directory)
        self.manifest = json.loads((directory / 'manifest.json').read_text())
        if self.manifest['settings'] != index_settings(config):
            raise ValueError('index settings differ from this experiment; rebuild or select the matching bundle')
        self.chunks = read_jsonl(directory / 'chunk_map.jsonl')
        cpu = faiss.read_index(str(directory / 'index.faiss'))
        if cpu.ntotal != len(self.chunks) or cpu.d != config['embedding']['dimensions']:
            raise ValueError('index and chunk map dimensions/counts disagree')
        self.resources = faiss.StandardGpuResources()
        self.resources.setTempMemory(GPU_TEMP_MEMORY_MIB * 1024 ** 2)
        # GPU top-K kernels have bounded K; small shards allow exact all-chunk scores.
        self.shards = []
        for offset in range(0, cpu.ntotal, 1024):
            vectors = cpu.reconstruct_n(offset, min(1024, cpu.ntotal - offset))
            shard = faiss.GpuIndexFlatIP(self.resources, cpu.d)
            shard.add(vectors)
            self.shards.append((offset, shard))

    def search(self, vector):
        # Exact all-chunk ranking also supplies the independent parent@100 ranking.
        hits = []
        for offset, shard in self.shards:
            scores, positions = shard.search(vector, shard.ntotal)
            hits.extend((self.chunks[offset + int(i)], float(score))
                        for i, score in zip(positions[0], scores[0]) if i >= 0)
        hits.sort(key=lambda hit: hit[1], reverse=True)
        return hits
