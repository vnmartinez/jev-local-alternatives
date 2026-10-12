# jev-local-alternatives

Notes on **Jev**, TypeSafe AI's "System One" decision model, plus a runnable example of an open-source model you can run locally for the same kind of task.

> Research snapshot from October 2026. Jev performance and adoption figures are **self-reported by TypeSafe**. No independent benchmark or architecture paper existed at the time of writing.

## What Jev is

- A **discriminative** model. It does not generate text. You give it unstructured context and typed questions (boolean, choice, ordered scale), and it returns a probability distribution with a confidence score for each.
- Labels are supplied at inference time (zero-shot), so it needs no task-specific training.
- Reportedly trained only on synthetic data with an unpublished method (RLCD) that targets **calibrated** probabilities. TypeSafe says it is transformer-based. Weights are closed.
- Released 15 Sep 2026 as a **hosted API only**. There is no self-hosting.
- Listed pricing: US$0.042 per 1M input tokens, output free. Claimed latency is 70–500 ms. Input is limited to 64k tokens.
- Questions in a single request are answered in parallel, so dependent questions need sequential calls.
- Competitor: OpenAI Decisions API (Oct 2026).
- Beware of look-alike sites. A "Jev AI" site exists that is not affiliated with TypeSafe.
- The name refers to the Jevons paradox: cheaper decisions should lead to more of them being automated.

## Can it run locally?

**No.** Jev is closed and hosted. The closest local option is an open-source zero-shot classifier:

| Model | License | Notes |
|---|---|---|
| [GLiClass v3](https://huggingface.co/knowledgator/gliclass-modern-base-v3.0) (Knowledgator) | Apache-2.0 | All labels in one forward pass; scales to many labels |
| [ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0) | see model card | NLI-style, 8k context |
| [DeBERTa-v3-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) | MIT | NLI-style; one pass per label, so it slows down with many labels |
| Small open LLM (Qwen, Llama) with constrained output | varies | Use per-option log-probs as scores; flexible but slower |

**Key gap:** none of these promise calibrated probabilities, which is Jev's main selling point. If you need calibration, fit temperature scaling on a labeled set of your own.

## Run the example (CPU is fine)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python classify.py
```

`classify.py` answers a typed question (support-ticket routing) for a few sample texts, prints a score for each label, and reports latency.

Sample run on a laptop CPU (no GPU), after the model was loaded:

```
    51 ms  billing          ok   I was charged twice for my subscription this month.
    46 ms  bug report       ok   The export button crashes the app when the list is empty.
    43 ms  feature request  ok   It would be great to have a dark mode.
    45 ms  account access   ok   I can't log in, the password reset email never arrives.
```

Four toy examples are not a benchmark. Scores are not calibrated probabilities.

## Sources

- [Jev (AI model), Wikipedia](https://en.wikipedia.org/wiki/Jev_(AI_model))
- [Introducing System One Models & Jev, TypeSafe AI blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [Fortune, 8 Oct 2026](https://fortune.com/2026/10/08/jev-an-ai-for-making-quick-decisions-has-been-a-viral-hit-in-silicon-valley-but-openai-is-hot-on-its-heels/)
- [TrueFoundry: What "System One Models" actually are](https://www.truefoundry.com/blog/typesafe-ai-jev)
- [GLiClass on PyPI](https://pypi.org/project/gliclass)

## License

MIT
