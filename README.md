# ADR causality eval (WHO-UMC)

Single-file harness: `adr_eval.py` (numpy only; `anthropic` / `openai` optional).

## Quick start
```bash
python adr_eval.py demo-cases --n 60 --out demo_cases.jsonl     # SYNTHETIC cases, pipeline smoke test only
python adr_eval.py run   --cases demo_cases.jsonl --models mock --out results.jsonl
python adr_eval.py score --cases demo_cases.jsonl --results results.jsonl --outdir report
```
`mock` is a noisy oracle for testing the pipeline. Its numbers mean nothing.

## Real run
```bash
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
python adr_eval.py run --cases my_cases.jsonl \
  --models anthropic:<model-id> openai:<model-id> \
  --conditions zero_shot cot structured --variants 0 1 2 --runs 3 --temperature 0
```
- Resumable: rerun the same command and finished jobs are skipped.
- A biomedical or local model served through an OpenAI-compatible endpoint: set `OPENAI_BASE_URL` and use `openai:<name>`.
- If an API rejects `temperature`, the Anthropic adapter retries without it. The `temperature` field in each result row records what was requested, so check the provider's actual behavior and log it in the report.

## Case file (JSONL)
Required: `case_id, report_text, drug, event, label_expert`.
Recommended: `cluster` (drug, for clustered bootstrap), `n_suspects`, `criteria` (expert criterion annotations, needed for criterion-level scoring),
`adversarial` + `pair_id` (matched control), `expert_ambiguous`, `post_cutoff`, `expert_a`/`expert_b` (inter-expert agreement), `label_info` (SmPC text; use `--provide-label-info`).

## What `score` reports
Per model x condition x prompt-wording: per-class recall/precision/F1, confusion matrix, accuracy, Cohen kappa, quadratic weighted kappa, Gwet AC1 (cluster-bootstrap CIs),
asymmetric and symmetric cost, downgrade/upgrade rates, off-by-k, strata (adversarial, expert-ambiguous, polypharmacy, post-cutoff), run-to-run flips and Fleiss kappa,
criterion accuracy, right-label-wrong-reason rate, adversarial vs matched-control failure (exact McNemar), failure typology, cross-prompt label changes, and a hypothesis tally (H1-H5).

## Caveats
- Metrics use the modal label across runs; flip rate and Fleiss kappa describe the runs themselves.
- The failure typology is heuristic (rule-based). Have two humans code a sample and report kappa before citing it.
- `implied_label` is an approximation of WHO-UMC logic, used only to flag reasoning/label contradictions.
- Set H3/H4/H5 thresholds (`--x-prompt-change --y-rlwr --z-flip`) from a pilot, and freeze them before the main run.
- Override the cost matrix with `--cost-matrix file.json` and report results under both cost versions.
