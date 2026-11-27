# Hero Dataset

Hero Dataset is a question-answering benchmark over knowledge graphs, tables, and infoboxes.

## Dataset

Version 2.0.0 contains **500 human-confirmed questions**, comprising 245 Movie and 255 TV Series questions, with 665 answer items. Gold annotations identify 2,291 evidence uses and 2,222 distinct evidence references.

| Collection | Complete records |
| --- | ---: |
| Golden Evidence parents | 1,702 |
| Negative Sample 1 | 209 |
| Negative Sample 2 | 20,582 |
| Negative Sample 3 | 162,363 |
| **Complete data repository** | **184,856** |

The complete repository combines Golden Evidence and all three Negative Samples: 145,115 KG records, 17,821 tables, and 21,920 infoboxes. Only the precise units listed in the Gold annotations are labeled Gold. Separate Negative Samples allow users to compose different difficulty variants.

## Naive RAG baseline

Naive RAG uses **BAAI/bge-m3** dense retrieval and **unsloth/Qwen3.8-27B-unsloth-bnb-4bit** generation through Transformers with NF4 weights and BF16 compute. Every machine uses the same implementation and checkpoint. Set paths and context/output budgets in `configs/runtime.yml`; see the [baseline instructions](baseline/NavReg/README.md) for setup and commands.

## Downloads

Download the four files from the [v2.0.0 Release](https://github.com/sby854905420-netizen/Hero-Dataset/releases/tag/v2.0.0). Extract the selected archives from the repository root; a Git clone does not include these corpus files.

| File | Size | Extraction location |
| --- | ---: | --- |
| [data_repository.tar.gz](https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/data_repository.tar.gz) | 109,439,171 bytes (104.37 MiB) | `Data/data_repository/` and `Data/source_mapping.jsonl` |
| [negative_sample_1.tar.gz](https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/negative_sample_1.tar.gz) | 256,372 bytes (0.24 MiB) | `Data/negative_samples/negative_sample_1/` |
| [negative_sample_2.tar.gz](https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/negative_sample_2.tar.gz) | 28,702,177 bytes (27.37 MiB) | `Data/negative_samples/negative_sample_2/` |
| [negative_sample_3.tar.gz](https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/negative_sample_3.tar.gz) | 69,272,493 bytes (66.06 MiB) | `Data/negative_samples/negative_sample_3/` |

```bash
tar -xzf data_repository.tar.gz
```

The complete repository already includes all Negative Samples; their separate archives are optional. Questions and confirmed answers are in `Data/qa_pairs.jsonl`, and Gold evidence annotations are in `Data/gold_annotations.jsonl`.

See [data formats](Data/README.md), [complete corpus](Data/data_repository/README.md), [Negative Samples](Data/negative_samples/README.md), [baseline](baseline/NavReg/README.md), and [evaluation](evaluation/README.md) for details. See [LICENSE](LICENSE), [data licensing](Data/LICENSE.md), and [CITATION.cff](CITATION.cff) for licensing and citation information.
