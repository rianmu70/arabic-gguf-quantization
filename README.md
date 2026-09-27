# Does Quantization Hurt Arabic More?

Code and data for the paper *"Does Quantization Hurt Arabic More? A Chance-Normalized, Paired Evaluation of GGUF-Quantized Small Language Models"* (R. A. Mubarak, submitted to JJCIT).

## Contents

| Path | Content |
|---|---|
| `notebooks/arabic_quantization_FINAL.ipynb` | Full pipeline: GGUF file discovery, evaluation with lm-evaluation-harness, sanity checks, paired bootstrap statistics with Holm correction |
| `analysis/make_figures.py` | Reproduces Figures 1-3 of the paper |
| `data/paper_tables/` | Summary tables used in the paper (RQ1, RQ2) |
| `results/` | Raw Kaggle outputs (item-level predictions, accuracies, tokenizer fertility) |

## Setup

- Models: Qwen3-4B, Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct (GGUF files by `mradermacher` on Hugging Face)
- Levels: F16, Q8_0, Q6_K, Q4_K_M, Q3_K_M, Q2_K (static) and Q6_K to Q2_K (imatrix)
- Benchmarks: Belebele (`arb_Arab`, `eng_Latn`), Global-MMLU-Lite (`ar`, `en`); zero-shot log-likelihood
- Statistics: item-level paired bootstrap (10,000 resamples), Holm correction, chance-normalized loss `L = 100 * (acc_F16 - acc_q) / (acc_F16 - 0.25)`

## Reproduce

1. Import `notebooks/arabic_quantization_FINAL.ipynb` into a new Kaggle notebook.
2. Settings: GPU T4 x2, Internet on. Run all (about 6 hours). CSV outputs go to `/kaggle/working`.
3. Figures: `cd analysis && python make_figures.py`

## License

MIT
