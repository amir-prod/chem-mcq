# Misconceptions corpus for RAG

After running misconception extraction, symlink or copy the corpus here:

```bash
ln -sf ../../../misconception_extraction/outputs/misconceptions_corpus.md \
  data/mds_misconceptions/misconceptions_corpus.md
```

Then rebuild the vector index:

```bash
python generate_mcqs.py --index-only --rebuild-index
```
