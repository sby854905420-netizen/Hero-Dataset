"""Official dense CLS embeddings on GPU; no silent model truncation or fallback."""
import numpy as np


class Embedder:
    def __init__(self, config, cache_dir=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for this embedding protocol')
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['revision'], cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(config['model_id'], revision=config['revision'], cache_dir=cache_dir,
                                               dtype=torch.float32).to('cuda').eval()
        self.batch_size = config['batch_size']
        if self.batch_size < 1:
            raise ValueError('embedding batch_size must be positive')

    def encode(self, texts, query=False):
        import torch
        prefix = self.config['query_prefix' if query else 'passage_prefix']
        batches = []
        for offset in range(0, len(texts), self.batch_size):
            values = [prefix + text for text in texts[offset:offset + self.batch_size]]
            tokens = self.tokenizer(values, padding=True, truncation=False, return_tensors='pt')
            if tokens['input_ids'].shape[1] > self.config['max_input_tokens']:
                raise ValueError('embedding input exceeds the registered native-token budget')
            with torch.inference_mode():
                result = self.model(**{k: v.to('cuda') for k, v in tokens.items()}).last_hidden_state[:, 0].float()
                norms = result.norm(dim=1, keepdim=True)
                if not torch.isfinite(result).all() or (norms == 0).any():
                    raise ValueError('non-finite or zero embedding vector')
                vectors = (result / norms).cpu().numpy()
            batches.append(vectors)
        vectors = np.concatenate(batches) if batches else np.empty((0, self.config['dimensions']), dtype=np.float32)
        if vectors.shape[1] != self.config['dimensions']:
            raise ValueError('embedding dimension differs from registered model')
        return np.ascontiguousarray(vectors, dtype=np.float32)
