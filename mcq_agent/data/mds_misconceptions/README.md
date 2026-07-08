# Misconceptions corpus for RAG

The agent indexes three topic-specific misconception corpora under this directory:

```text
data/mds_misconceptions/
├── mds_IMFs/                  -> ../../../dataFolder/mds_IMFs
├── mds_ox_redox/              -> ../../../dataFolder/mds_ox_redox
└── mds_sn1_sn2_reduction/     -> ../../../dataFolder/mds_sn1_sn2_reduction
```

Each topic folder contains:

- `misconceptions_corpus.md` — human-readable misconception entries
- `extracted_misconceptions.json` — structured extraction output

Rebuild the vector index after adding or updating corpora:

```bash
python generate_mcqs.py --index-only --rebuild-index
```
