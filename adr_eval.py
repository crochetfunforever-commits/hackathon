#!/usr/bin/env python3
"""
ADR causality-assessment evaluation harness (WHO-UMC primary).

Subcommands
  demo-cases   write a synthetic case file for smoke-testing the pipeline
  run          query models over cases x conditions x variants x repeats (resumable)
  score        compute metrics, failure analyses, stability, hypothesis summary

Case file (JSONL), one object per line:
  case_id, report_text, drug, event, label_expert (Certain|Probable|Possible|Unlikely|Conditional)
  optional: cluster (defaults to drug), stratum, n_suspects, adversarial (bool), pair_id (id of matched
  control), expert_ambiguous (bool), post_cutoff (bool), label_info (text), expert_a/expert_b (labels),
  criteria {event_listed, temporal_plausible, alternative_causes: yes|no|unclear;
            dechallenge, rechallenge: positive|negative|not_done|unknown}
"""
import argparse, json, re, os, sys, math, random, time, hashlib, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

LABELS = ["Certain", "Probable", "Possible", "Unlikely", "Conditional"]
ORD = LABELS[:4]                       # ordinal categories, most -> least certain
ALL = LABELS + ["INVALID"]
IDX = {l: i for i, l in enumerate(ALL)}
CRIT = ["event_listed", "temporal_plausible", "alternative_causes", "dechallenge", "rechallenge"]

# Cost[expert][model]. Downgrades of strong causality are costliest; upgrades also penalised.
DEFAULT_COST = {
    "Certain":     {"Certain": 0, "Probable": 2, "Possible": 5, "Unlikely": 10, "Conditional": 6},
    "Probable":    {"Certain": 1, "Probable": 0, "Possible": 2, "Unlikely": 6,  "Conditional": 4},
    "Possible":    {"Certain": 2, "Probable": 1, "Possible": 0, "Unlikely": 2,  "Conditional": 2},
    "Unlikely":    {"Certain": 4, "Probable": 3, "Possible": 1, "Unlikely": 0,  "Conditional": 2},
    "Conditional": {"Certain": 3, "Probable": 2, "Possible": 1, "Unlikely": 1,  "Conditional": 0},
}
def symmetric_cost():
    c = {}
    for a in LABELS:
        c[a] = {}
        for b in LABELS:
            c[a][b] = abs(ORD.index(a) - ORD.index(b)) if a in ORD and b in ORD else (0 if a == b else 2)
    return c

# ----------------------------------------------------------------------------- prompts
WHO_UMC = """WHO-UMC causality categories:
- Certain: plausible time relationship; cannot be explained by disease or other drugs; response to withdrawal plausible; event definitive pharmacologically/phenomenologically; rechallenge satisfactory if necessary.
- Probable: reasonable time relationship; unlikely to be attributed to disease or other drugs; clinically reasonable response on withdrawal; rechallenge not required.
- Possible: reasonable time relationship; could also be explained by disease or other drugs; information on withdrawal may be lacking or unclear.
- Unlikely: time relationship makes a causal link improbable; disease or other drugs provide plausible explanations.
- Conditional: more data are needed for proper assessment (Conditional/Unclassifiable)."""

VARIANT_INTRO = [
    "You are an experienced pharmacovigilance assessor. Assess causality between the named suspect drug and the event using the WHO-UMC system.",
    "Act as a clinical pharmacologist performing a WHO-UMC causality assessment of one drug-event pair from an adverse event report.",
    "Your task: WHO-UMC causality classification for the specified drug-event pair, as a drug-safety physician would do it.",
]

def build_prompt(case, condition, variant, provide_label_info=False):
    intro = VARIANT_INTRO[variant % len(VARIANT_INTRO)]
    body = f"{WHO_UMC}\n\nCASE REPORT:\n{case['report_text']}\n\nSUSPECT DRUG: {case['drug']}\nEVENT: {case['event']}\n"
    if provide_label_info and case.get("label_info"):
        body += f"\nPRODUCT LABEL INFORMATION:\n{case['label_info']}\n"
    cats = "Certain, Probable, Possible, Unlikely, Conditional"
    if condition == "zero_shot":
        tail = f"\nRespond with only one line: `LABEL: <one of {cats}>`."
    elif condition == "cot":
        tail = f"\nThink step by step, then end with a final line `FINAL LABEL: <one of {cats}>`."
    elif condition == "structured":
        tail = ("\nAddress each criterion explicitly BEFORE choosing the label. Return ONLY a JSON object with keys:\n"
                '"event_listed": yes|no|unclear  (is the event a known reaction of the drug?),\n'
                '"temporal_plausible": yes|no|unclear,\n'
                '"alternative_causes": yes|no|unclear  (are there plausible alternative explanations?),\n'
                '"dechallenge": positive|negative|not_done|unknown,\n'
                '"rechallenge": positive|negative|not_done|unknown,\n'
                '"rationale": short string,\n'
                f'"label": one of {cats}.')
    else:
        raise ValueError(condition)
    return intro, body + tail

# ----------------------------------------------------------------------------- parsing
def normalize_label(s):
    if s is None: return None
    t = str(s).lower()
    for pat, lab in [(r"conditional|unclassif|unassess", "Conditional"), (r"unlikely|doubtful", "Unlikely"),
                     (r"certain|definite", "Certain"), (r"probable|likely", "Probable"), (r"possible", "Possible")]:
        if re.search(pat, t): return lab
    return None

def norm_val(v):
    if v is None: return None
    t = re.sub(r"[\s\-]+", "_", str(v).strip().lower())
    return {"not_done": "not_done", "n/a": "unknown", "na": "unknown", "unknown": "unknown",
            "uncertain": "unclear", "true": "yes", "false": "no"}.get(t, t)

def crit_equal(k, a, b):
    a, b = norm_val(a), norm_val(b)
    if k in ("event_listed", "temporal_plausible", "alternative_causes"):
        a = "unclear" if a in ("unknown", None) else a
        b = "unclear" if b in ("unknown", None) else b
    return a == b

def extract_json(text):
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i: return None
    try: return json.loads(text[i:j + 1])
    except Exception: return None

def parse_output(text, condition):
    label, crit = None, None
    if condition == "structured":
        j = extract_json(text or "")
        if isinstance(j, dict):
            label = normalize_label(j.get("label"))
            crit = {k: norm_val(j.get(k)) for k in CRIT}
    else:
        m = re.findall(r"LABEL\s*[:=]\s*\**\s*([^\n*]+)", text or "", re.I)
        if m: label = normalize_label(m[-1])
        elif text and len(text.strip()) < 40: label = normalize_label(text)
    return label or "INVALID", crit

# ----------------------------------------------------------------------------- models
class MockModel:
    """Noisy oracle for pipeline smoke tests ONLY. Results are meaningless."""
    def __init__(self, name="mock"): self.name = name
    def complete(self, system, user, meta):
        c, cond = meta["case"], meta["condition"]
        rng = random.Random(f"{c['case_id']}|{cond}|{meta['variant']}|{meta['run']}")
        gold = c["label_expert"]
        pc = 0.4 if c.get("adversarial") else (0.45 if gold == "Possible" else 0.7)
        if rng.random() < 0.03: return "I cannot provide medical advice on this."
        lab = gold
        if rng.random() > pc and gold in ORD:
            i = ORD.index(gold)
            lab = ORD[max(0, min(3, i + rng.choice([-1, 1])))]
        crit = dict(c.get("criteria") or {})
        for k in CRIT:
            if rng.random() < 0.15 or (c.get("adversarial") and k == "rechallenge" and rng.random() < 0.5):
                crit[k] = "not_done" if k in ("dechallenge", "rechallenge") else "unclear"
        if cond == "zero_shot": return f"LABEL: {lab}"
        if cond == "cot": return f"Reasoning...\nFINAL LABEL: {lab}"
        return json.dumps({**crit, "rationale": "mock", "label": lab})

class AnthropicModel:
    def __init__(self, name, temperature, max_tokens):
        import anthropic
        self.name, self.t, self.mt, self.client = name, temperature, max_tokens, anthropic.Anthropic()
        self.temp_ok = True
    def complete(self, system, user, meta):
        kw = dict(model=self.name, max_tokens=self.mt, system=system, messages=[{"role": "user", "content": user}])
        if self.temp_ok: kw["temperature"] = self.t
        try: r = self.client.messages.create(**kw)
        except Exception as e:
            if self.temp_ok and "temperature" in str(e).lower():
                self.temp_ok = False; kw.pop("temperature"); r = self.client.messages.create(**kw)
            else: raise
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

class OpenAIModel:
    """Also serves any OpenAI-compatible endpoint (vLLM, etc.) via OPENAI_BASE_URL."""
    def __init__(self, name, temperature, max_tokens):
        import openai
        self.name, self.t, self.mt, self.client = name, temperature, max_tokens, openai.OpenAI()
    def complete(self, system, user, meta):
        r = self.client.chat.completions.create(model=self.name, temperature=self.t, max_tokens=self.mt,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content or ""

def make_model(spec, temperature, max_tokens):
    if spec.startswith("mock"): return MockModel(spec)
    prov, _, name = spec.partition(":")
    if prov == "anthropic": return AnthropicModel(name, temperature, max_tokens)
    if prov == "openai": return OpenAIModel(name, temperature, max_tokens)
    raise ValueError(f"unknown model spec {spec!r}; use mock | anthropic:<model> | openai:<model>")

# ----------------------------------------------------------------------------- run
def load_jsonl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]

def cmd_run(a):
    cases = load_jsonl(a.cases)
    done = set()
    if os.path.exists(a.out):
        for r in load_jsonl(a.out):
            done.add((r["case_id"], r["model"], r["condition"], r["variant"], r["run"]))
    models = {s: make_model(s, a.temperature, a.max_tokens) for s in a.models}
    jobs = [(c, s, cond, v, run) for c in cases for s in a.models for cond in a.conditions
            for v in a.variants for run in range(a.runs) if (c["case_id"], s, cond, v, run) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done)")
    lock, t0 = threading.Lock(), time.time()
    fout = open(a.out, "a")

    def work(job):
        c, s, cond, v, run = job
        system, user = build_prompt(c, cond, v, a.provide_label_info)
        raw, err = "", None
        for attempt in range(4):
            try: raw = models[s].complete(system, user, {"case": c, "condition": cond, "variant": v, "run": run}); err = None; break
            except Exception as e: err = repr(e)[:300]; time.sleep(2 ** attempt)
        label, crit = parse_output(raw, cond) if not err else ("INVALID", None)
        return {"case_id": c["case_id"], "model": s, "condition": cond, "variant": v, "run": run,
                "label": label, "criteria": crit, "raw": raw, "error": err, "temperature": a.temperature}

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            with lock:
                fout.write(json.dumps(f.result()) + "\n"); fout.flush()
            if n % 50 == 0: print(f"  {n}/{len(jobs)}  {time.time() - t0:.0f}s")
    fout.close(); print("done ->", a.out)

# ----------------------------------------------------------------------------- metrics
def confusion(pairs):
    m = np.zeros((len(ALL), len(ALL)), int)
    for t, p in pairs: m[IDX[t], IDX[p]] += 1
    return m

def kappa_from(m, d):
    n = m.sum()
    if n == 0: return float("nan")
    O, E = m / n, np.outer(m.sum(1), m.sum(0)) / n ** 2
    den = (d * E).sum()
    if den == 0: return 1.0 if (d * O).sum() == 0 else float("nan")
    return 1 - (d * O).sum() / den

def cohen_kappa(pairs):
    m = confusion(pairs); return kappa_from(m, 1 - np.eye(len(ALL)))

def qw_kappa(pairs):
    pp = [(ORD.index(t), ORD.index(p)) for t, p in pairs if t in ORD and p in ORD]
    if not pp: return float("nan")
    m = np.zeros((4, 4), int)
    for t, p in pp: m[t, p] += 1
    d = (np.arange(4)[:, None] - np.arange(4)[None, :]) ** 2.0
    return kappa_from(m, d)

def gwet_ac1(pairs):
    m = confusion(pairs); n = m.sum()
    if n == 0: return float("nan")
    q = len(ALL); pa = np.trace(m) / n
    pi = (m.sum(1) + m.sum(0)) / (2 * n); pe = (pi * (1 - pi)).sum() / (q - 1)
    return (pa - pe) / (1 - pe) if pe < 1 else float("nan")

def cost_of(C, t, p):
    return C[t][p] if p in C[t] else max(C[t].values())

def summarize(pairs, C, Csym, full=True):
    n = len(pairs)
    if n == 0: return {}
    acc = sum(t == p for t, p in pairs) / n
    poss = [p for t, p in pairs if t == "Possible"]
    ordp = [(t, p) for t, p in pairs if t in ORD and p in ORD]
    down = sum(ORD.index(p) > ORD.index(t) for t, p in ordp)
    up = sum(ORD.index(p) < ORD.index(t) for t, p in ordp)
    s = {"n": n, "acc": acc, "qwk": qw_kappa(pairs),
         "mean_cost": float(np.mean([cost_of(C, t, p) for t, p in pairs])),
         "possible_recall": (sum(p == "Possible" for p in poss) / len(poss)) if poss else float("nan"),
         "downgrade_rate": down / len(ordp) if ordp else float("nan")}
    if full:
        s.update({"kappa": cohen_kappa(pairs), "ac1": gwet_ac1(pairs),
                  "mean_cost_symmetric": float(np.mean([cost_of(Csym, t, p) for t, p in pairs])),
                  "upgrade_rate": up / len(ordp) if ordp else float("nan"),
                  "off_by_1": sum(abs(ORD.index(p) - ORD.index(t)) == 1 for t, p in ordp) / len(ordp) if ordp else float("nan"),
                  "off_by_2plus": sum(abs(ORD.index(p) - ORD.index(t)) >= 2 for t, p in ordp) / len(ordp) if ordp else float("nan"),
                  "invalid_rate": sum(p == "INVALID" for _, p in pairs) / n})
    return s

def per_class(m):
    out = {}
    for l in LABELS:
        i = IDX[l]; tp = m[i, i]; r, c = m[i].sum(), m[:, i].sum()
        rec = tp / r if r else float("nan"); pre = tp / c if c else float("nan")
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) and not math.isnan(pre + rec) else float("nan")
        out[l] = {"support": int(r), "recall": rec, "precision": pre, "f1": f1}
    return out

def cluster_bootstrap(cl_pairs, C, Csym, B, seed=0, keys=("acc", "qwk", "mean_cost", "possible_recall", "downgrade_rate")):
    rng = random.Random(seed); ks = list(cl_pairs)
    res = {k: [] for k in keys}
    for _ in range(B):
        pairs = [x for k in (rng.choice(ks) for _ in ks) for x in cl_pairs[k]]
        s = summarize(pairs, C, Csym, full=False)
        for k in keys: res[k].append(s[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if not np.all(np.isnan(v)) else [float("nan")] * 2
            for k, v in res.items()}

def fleiss_kappa(rows):
    """rows: list of lists of labels (equal length R)."""
    N, R = len(rows), len(rows[0])
    M = np.zeros((N, len(ALL)))
    for i, r in enumerate(rows):
        for l in r: M[i, IDX[l]] += 1
    Pi = ((M ** 2).sum(1) - R) / (R * (R - 1)); Pbar = Pi.mean()
    pj = M.sum(0) / (N * R); Pe = (pj ** 2).sum()
    if Pe >= 1: return 1.0 if Pbar >= 1 else float("nan")
    return float((Pbar - Pe) / (1 - Pe))

def binom_two_sided(k, n):
    if n == 0: return float("nan")
    p = sum(math.comb(n, i) for i in range(n + 1) if math.comb(n, i) <= math.comb(n, k)) / 2 ** n
    return min(1.0, p)

def implied_label(c):
    """Label implied by a model's own criterion judgments (approximate WHO-UMC logic)."""
    if c["temporal_plausible"] == "no": return "Unlikely"
    if c["temporal_plausible"] in ("unclear", "unknown", None): return "Conditional"
    if c["alternative_causes"] != "no": return "Possible"
    if c["dechallenge"] == "positive": return "Certain" if c["rechallenge"] == "positive" else "Probable"
    return "Possible"

def classify_failure(row, case, modal_pred):
    """Priority-ordered typology of a wrong run-level prediction."""
    if row["label"] == "INVALID": return "irrelevant_or_evasive"
    crit, gold = row.get("criteria"), case.get("criteria") or {}
    if crit and all(crit.get(k) for k in CRIT):
        if implied_label(crit) != row["label"]: return "reasoning_label_contradiction"
    if crit and any(gold.get(k) == "positive" and crit.get(k) != "positive" for k in ("dechallenge", "rechallenge")):
        return "omission_of_decisive_criterion"
    if row["label"] == "Certain" and case["label_expert"] != "Certain": return "spurious_certainty"
    if row["label"] == modal_pred: return "collapse_to_modal_label"
    return "other"

# ----------------------------------------------------------------------------- score
def modal(labels):
    c = Counter(labels); top = max(c.values())
    return next(l for l in labels if c[l] == top)      # ties -> earliest run

def fmt(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{d}f}" if isinstance(x, float) else str(x))

def cmd_score(a):
    cases = {c["case_id"]: c for c in load_jsonl(a.cases)}
    rows = [r for r in load_jsonl(a.results) if r["case_id"] in cases]
    C = json.load(open(a.cost_matrix)) if a.cost_matrix else DEFAULT_COST
    Csym = symmetric_cost()
    os.makedirs(a.outdir, exist_ok=True)
    report, lines = {"expert": {}, "groups": {}, "cross": {}, "hypotheses": {}}, []
    P = lines.append

    # --- expert reference quality
    ea = [(c["expert_a"], c["expert_b"]) for c in cases.values() if c.get("expert_a") and c.get("expert_b")]
    P("=" * 78); P("DATASET / REFERENCE STANDARD"); P("=" * 78)
    P(f"cases: {len(cases)}   label distribution: {dict(Counter(c['label_expert'] for c in cases.values()))}")
    P(f"adversarial: {sum(bool(c.get('adversarial')) for c in cases.values())}   "
      f"expert-ambiguous: {sum(bool(c.get('expert_ambiguous')) for c in cases.values())}   "
      f"polypharmacy(>=2 suspects): {sum((c.get('n_suspects') or 1) >= 2 for c in cases.values())}")
    if ea:
        report["expert"] = {"n": len(ea), "agreement": sum(x == y for x, y in ea) / len(ea),
                            "kappa": cohen_kappa(ea), "qwk": qw_kappa(ea), "ac1": gwet_ac1(ea)}
        P("expert A vs B: " + "  ".join(f"{k}={fmt(v)}" for k, v in report["expert"].items()))
        P("(compare model-vs-expert agreement below against this ceiling)")

    groups = defaultdict(list)
    for r in rows: groups[(r["model"], r["condition"], r["variant"])].append(r)
    modal_tab = {}                                         # (model,cond,variant) -> {case: label}
    for g, rs in groups.items():
        by = defaultdict(list)
        for r in sorted(rs, key=lambda r: r["run"]): by[r["case_id"]].append(r["label"])
        modal_tab[g] = {k: modal(v) for k, v in by.items()}

    for g in sorted(groups):
        model, cond, var = g; rs = groups[g]; mt = modal_tab[g]
        pairs = [(cases[k]["label_expert"], p) for k, p in mt.items()]
        cl = defaultdict(list)
        for k, p in mt.items():
            cl[cases[k].get("cluster") or cases[k].get("drug") or k].append((cases[k]["label_expert"], p))
        S = summarize(pairs, C, Csym); ci = cluster_bootstrap(cl, C, Csym, a.boot)
        m = confusion(pairs); pc = per_class(m)
        G = {"summary": S, "ci95": ci, "per_class": pc, "confusion": m.tolist()}

        P(""); P("=" * 78); P(f"MODEL={model}  CONDITION={cond}  PROMPT_VARIANT={var}   (n_cases={S['n']}, label=modal over runs)"); P("=" * 78)
        P("per-class (expert label -> recall/precision/F1):")
        for l in LABELS:
            x = pc[l]; P(f"  {l:<12} n={x['support']:<4} R={fmt(x['recall'])}  P={fmt(x['precision'])}  F1={fmt(x['f1'])}")
        P("confusion (rows=expert, cols=model):  " + " ".join(f"{l[:5]:>6}" for l in ALL))
        for l in LABELS: P(f"  {l:<12}              " + " ".join(f"{v:>6}" for v in m[IDX[l]]))
        P(f"acc={fmt(S['acc'])} [{fmt(ci['acc'][0])},{fmt(ci['acc'][1])}]  kappa={fmt(S['kappa'])}  "
          f"QWK={fmt(S['qwk'])} [{fmt(ci['qwk'][0])},{fmt(ci['qwk'][1])}]  AC1={fmt(S['ac1'])}")
        P(f"cost(asym)={fmt(S['mean_cost'])} [{fmt(ci['mean_cost'][0])},{fmt(ci['mean_cost'][1])}]  cost(sym)={fmt(S['mean_cost_symmetric'])}  "
          f"downgrade={fmt(S['downgrade_rate'])}  upgrade={fmt(S['upgrade_rate'])}  "
          f"off-by-1={fmt(S['off_by_1'])}  off-by-2+={fmt(S['off_by_2plus'])}  invalid={fmt(S['invalid_rate'])}")

        # strata
        strata = {"adversarial": lambda c: bool(c.get("adversarial")), "expert_ambiguous": lambda c: bool(c.get("expert_ambiguous")),
                  "polypharmacy": lambda c: (c.get("n_suspects") or 1) >= 2, "post_cutoff": lambda c: bool(c.get("post_cutoff"))}
        G["strata"] = {}
        for sname, fn in strata.items():
            for val in (True, False):
                sp = [(cases[k]["label_expert"], p) for k, p in mt.items() if fn(cases[k]) == val]
                if sp and (val or sname != "post_cutoff"):
                    s2 = summarize(sp, C, Csym, full=False); G["strata"][f"{sname}={val}"] = s2
            if f"{sname}=True" in G["strata"]:
                t, f = G["strata"][f"{sname}=True"], G["strata"].get(f"{sname}=False", {})
                P(f"  stratum {sname:<17} True: n={t['n']} acc={fmt(t['acc'])} cost={fmt(t['mean_cost'])} | False: acc={fmt(f.get('acc', float('nan')))} cost={fmt(f.get('mean_cost', float('nan')))}")

        # stability across runs
        by = defaultdict(list)
        for r in sorted(rs, key=lambda r: r["run"]): by[r["case_id"]].append(r["label"])
        R = min(len(v) for v in by.values())
        if R >= 2:
            fl = sum(len(set(v[:R])) > 1 for v in by.values()) / len(by)
            G["stability"] = {"runs": R, "flip_rate": fl, "fleiss_kappa": fleiss_kappa([v[:R] for v in by.values()])}
            P(f"stability: runs={R}  label-flip rate={fmt(fl)}  Fleiss kappa across runs={fmt(G['stability']['fleiss_kappa'])}")

        # criterion-level (structured only)
        if cond == "structured":
            crit_acc, rlwr_den, rlwr_num, n_crit = {}, 0, 0, 0
            for k in CRIT:
                ok = [crit_equal(k, r["criteria"].get(k), cases[r["case_id"]]["criteria"].get(k)) for r in rs
                      if r.get("criteria") and (cases[r["case_id"]].get("criteria") or {}).get(k) is not None]
                crit_acc[k] = sum(ok) / len(ok) if ok else float("nan")
            for r in rs:
                gc = cases[r["case_id"]].get("criteria")
                if not gc or r["label"] != cases[r["case_id"]]["label_expert"]: continue
                rlwr_den += 1
                if not r.get("criteria") or any(not crit_equal(k, r["criteria"].get(k), gc.get(k)) for k in CRIT if gc.get(k) is not None):
                    rlwr_num += 1
            G["criterion_accuracy"] = crit_acc
            G["right_label_wrong_reason"] = rlwr_num / rlwr_den if rlwr_den else float("nan")
            P("criterion accuracy: " + "  ".join(f"{k}={fmt(v)}" for k, v in crit_acc.items()))
            P(f"right-label-wrong-reason rate (among correct runs): {fmt(G['right_label_wrong_reason'])}")

        # adversarial vs matched control
        adv = [c for c in cases.values() if c.get("adversarial") and c.get("pair_id") in cases and c["case_id"] in mt and c["pair_id"] in mt]
        if adv:
            b = sum(mt[c["case_id"]] != c["label_expert"] and mt[c["pair_id"]] == cases[c["pair_id"]]["label_expert"] for c in adv)
            cc = sum(mt[c["case_id"]] == c["label_expert"] and mt[c["pair_id"]] != cases[c["pair_id"]]["label_expert"] for c in adv)
            fa = sum(mt[c["case_id"]] != c["label_expert"] for c in adv) / len(adv)
            fc = sum(mt[c["pair_id"]] != cases[c["pair_id"]]["label_expert"] for c in adv) / len(adv)
            G["adversarial"] = {"pairs": len(adv), "fail_adv": fa, "fail_control": fc, "adv_only_fail": b, "control_only_fail": cc,
                                "mcnemar_exact_p": binom_two_sided(b, b + cc)}
            P(f"adversarial pairs={len(adv)}  fail(adv)={fmt(fa)}  fail(matched control)={fmt(fc)}  discordant b={b} c={cc}  exact p={fmt(G['adversarial']['mcnemar_exact_p'])}")

        # failure typology (run-level)
        mp = Counter(p for p in mt.values()).most_common(1)[0][0]
        typ = Counter(classify_failure(r, cases[r["case_id"]], mp) for r in rs if r["label"] != cases[r["case_id"]]["label_expert"])
        G["failure_typology"] = dict(typ)
        P("failure typology (wrong runs): " + ", ".join(f"{k}={v}" for k, v in typ.most_common()))
        report["groups"]["|".join(map(str, g))] = G
        np.savetxt(os.path.join(a.outdir, f"confusion_{model.replace(':', '_')}_{cond}_v{var}.csv"), m, fmt="%d", delimiter=",",
                   header=",".join(ALL), comments="")

    # --- cross-condition / cross-variant instability
    P(""); P("=" * 78); P("CROSS-PROMPT INSTABILITY (modal labels)"); P("=" * 78)
    for model in sorted({g[0] for g in groups}):
        for var in sorted({g[2] for g in groups if g[0] == model}):
            for c1, c2 in [("zero_shot", "structured"), ("cot", "structured"), ("zero_shot", "cot")]:
                if (model, c1, var) in modal_tab and (model, c2, var) in modal_tab:
                    A, B = modal_tab[(model, c1, var)], modal_tab[(model, c2, var)]
                    ks = [k for k in A if k in B]
                    ag = sum(A[k] == B[k] for k in ks) / len(ks)
                    kp = cohen_kappa([(A[k], B[k]) for k in ks])
                    report["cross"][f"{model}|v{var}|{c1}~{c2}"] = {"agreement": ag, "change_rate": 1 - ag, "kappa": kp}
                    P(f"{model} v{var}: {c1} vs {c2}: label changes in {fmt(1 - ag)} of cases (kappa={fmt(kp)})")
        for cond in sorted({g[1] for g in groups if g[0] == model}):
            vs = sorted(g[2] for g in groups if g[0] == model and g[1] == cond)
            for i in range(len(vs) - 1):
                A, B = modal_tab[(model, cond, vs[i])], modal_tab[(model, cond, vs[i + 1])]
                ks = [k for k in A if k in B]
                P(f"{model} {cond}: prompt-wording v{vs[i]} vs v{vs[i+1]} agreement={fmt(sum(A[k]==B[k] for k in ks)/len(ks))}")

    # --- pre-registered hypotheses
    P(""); P("=" * 78); P(f"PRE-REGISTERED HYPOTHESES  (X={a.x_prompt_change}, Y={a.y_rlwr}, Z={a.z_flip})"); P("=" * 78)
    H = defaultdict(list)
    for g, G in report["groups"].items():
        pc = G["per_class"]; sup = {l: v["recall"] for l, v in pc.items() if l in ORD and v["support"] > 0 and not math.isnan(v["recall"])}
        if "Possible" in sup: H["H1_possible_lowest_recall"].append((g, sup["Possible"] == min(sup.values())))
        if "adversarial" in G: H["H2_adversarial_worse_than_control"].append((g, G["adversarial"]["fail_adv"] > G["adversarial"]["fail_control"] and G["adversarial"]["mcnemar_exact_p"] < 0.05))
        if G.get("right_label_wrong_reason") is not None and "right_label_wrong_reason" in G: H["H4_right_label_wrong_reason>Y"].append((g, G["right_label_wrong_reason"] > a.y_rlwr))
        if "stability" in G: H["H5_run_flip_rate>Z"].append((g, G["stability"]["flip_rate"] > a.z_flip))
    for k, v in report["cross"].items():
        if k.endswith("zero_shot~structured"): H["H3_zs_vs_structured_change>X"].append((k, v["change_rate"] > a.x_prompt_change))
    for h, res in sorted(H.items()):
        s = sum(x for _, x in res); P(f"{h:<36} supported in {s}/{len(res)} model-condition cells")
        report["hypotheses"][h] = {"supported": s, "cells": len(res), "detail": {k: bool(x) for k, x in res}}
    P("\nClaim rule: report 'failure' only where H1/H2/H4/H5 (+H3) are supported consistently across models and prompts;")
    P("otherwise describe the failure as model- or prompt-specific. Null results are findings.")

    text = "\n".join(lines); print(text)
    open(os.path.join(a.outdir, "report.txt"), "w").write(text)
    json.dump(report, open(os.path.join(a.outdir, "report.json"), "w"), indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))
    print(f"\nwrote {a.outdir}/report.txt, report.json, confusion_*.csv")

# ----------------------------------------------------------------------------- demo cases (synthetic)
def cmd_demo(a):
    rng = random.Random(1)
    pat = {"Certain": dict(event_listed="yes", temporal_plausible="yes", alternative_causes="no", dechallenge="positive", rechallenge="positive"),
           "Probable": dict(event_listed="yes", temporal_plausible="yes", alternative_causes="no", dechallenge="positive", rechallenge="not_done"),
           "Possible": dict(event_listed="yes", temporal_plausible="yes", alternative_causes="yes", dechallenge="unknown", rechallenge="not_done"),
           "Unlikely": dict(event_listed="no", temporal_plausible="no", alternative_causes="yes", dechallenge="negative", rechallenge="not_done"),
           "Conditional": dict(event_listed="unclear", temporal_plausible="unclear", alternative_causes="unclear", dechallenge="unknown", rechallenge="unknown")}
    def text(drug, ev, c, extra=""):
        t = [f"A {rng.randint(25, 80)}-year-old patient was started on {drug}."]
        t.append({"yes": f"{ev} began 10 days later.", "no": f"{ev} began 14 months after the drug was started.", "unclear": f"The onset of {ev} is not documented."}[c["temporal_plausible"]])
        t.append({"yes": "The patient has a documented comorbid condition and takes other drugs that can cause this event.", "no": "No other drugs or illness explain the event.", "unclear": "Medical history is not provided."}[c["alternative_causes"]])
        t.append({"positive": f"{ev} resolved after {drug} was stopped.", "negative": f"{ev} persisted after {drug} was stopped.", "unknown": "Outcome after withdrawal is unknown.", "not_done": ""}[c["dechallenge"]])
        t.append({"positive": f"{drug} was restarted and {ev} recurred.", "negative": f"{drug} was restarted without recurrence.", "not_done": "", "unknown": ""}[c["rechallenge"]])
        return " ".join(x for x in t if x) + (" " + extra if extra else "")
    out, i = [], 0
    per = max(1, a.n // 6)
    for lab, c in pat.items():
        for _ in range(per):
            i += 1; drug = f"DrugX{rng.randint(1, 8)}"; ev = rng.choice(["hepatitis", "rash", "neutropenia", "QT prolongation"])
            out.append(dict(case_id=f"c{i:03d}", drug=drug, event=ev, report_text=text(drug, ev, c), label_expert=lab, criteria=c,
                            n_suspects=rng.choice([1, 1, 2, 4]), cluster=drug, expert_a=lab, expert_b=lab if rng.random() > .2 else rng.choice(ORD),
                            expert_ambiguous=rng.random() < .1, stratum=lab))
    for k in range(per):                                    # adversarial / matched-control pairs (Certain with hedging noise)
        i += 1; drug = f"DrugY{k % 4}"; ev = "hepatitis"; c = pat["Certain"]
        ctrl = dict(case_id=f"c{i:03d}", drug=drug, event=ev, report_text=text(drug, ev, c), label_expert="Certain", criteria=c, n_suspects=1, cluster=drug, stratum="control")
        i += 1
        adv = dict(case_id=f"c{i:03d}", drug=drug, event=ev, report_text=text(drug, ev, c, "Reporter comment: relationship unclear, busy ward with many confounders."),
                   label_expert="Certain", criteria=c, n_suspects=1, cluster=drug, adversarial=True, pair_id=ctrl["case_id"], stratum="adversarial")
        out += [ctrl, adv]
    with open(a.out, "w") as f:
        for c in out: f.write(json.dumps(c) + "\n")
    print(f"wrote {len(out)} SYNTHETIC cases -> {a.out}")

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("demo-cases"); d.add_argument("--n", type=int, default=60); d.add_argument("--out", default="demo_cases.jsonl"); d.set_defaults(f=cmd_demo)
    r = sp.add_parser("run")
    r.add_argument("--cases", required=True); r.add_argument("--out", default="results.jsonl")
    r.add_argument("--models", nargs="+", required=True, help="mock | anthropic:<model> | openai:<model>")
    r.add_argument("--conditions", nargs="+", default=["zero_shot", "cot", "structured"], choices=["zero_shot", "cot", "structured"])
    r.add_argument("--variants", nargs="+", type=int, default=[0, 1, 2], help="prompt-wording variants 0-2")
    r.add_argument("--runs", type=int, default=3); r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-tokens", type=int, default=1500); r.add_argument("--workers", type=int, default=8)
    r.add_argument("--provide-label-info", action="store_true", help="include case['label_info'] (listed-status source) in the prompt")
    r.set_defaults(f=cmd_run)
    s = sp.add_parser("score")
    s.add_argument("--cases", required=True); s.add_argument("--results", default="results.jsonl"); s.add_argument("--outdir", default="report")
    s.add_argument("--cost-matrix", help="JSON {expert:{model:cost}} overriding the default"); s.add_argument("--boot", type=int, default=1000)
    s.add_argument("--x-prompt-change", type=float, default=0.15, help="H3 threshold X"); s.add_argument("--y-rlwr", type=float, default=0.20, help="H4 threshold Y")
    s.add_argument("--z-flip", type=float, default=0.05, help="H5 threshold Z"); s.set_defaults(f=cmd_score)
    a = ap.parse_args(); a.f(a)

if __name__ == "__main__":
    main()
