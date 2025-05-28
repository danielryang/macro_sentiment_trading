"""
FinBERT sentiment scoring pipeline.

- Uses ProsusAI/finbert via HuggingFace Transformers to score headlines.
- Requires HUGGINGFACE_TOKEN if rate-limited.
- Input: Cleaned headlines (data/processed/)
- Output: Sentiment scores (data/processed/)
- Reference: arXiv:2505.16136v1
"""
