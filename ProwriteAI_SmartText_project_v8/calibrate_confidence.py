"""Calibrate the app's confidence thresholds on validation.csv (never on test.csv).

For every validation message, the selected model generates an output with the app's own pipeline; we record the
mean log-probability of the generated tokens and the sentence-level chrF against the reference. Thresholds are chosen
so that 'high' outputs are almost always good and 'low' outputs catch as many bad outputs as possible.

Usage:  python calibrate_confidence.py [n_messages]
"""
import csv, os, sys, json
import sacrebleu

sys.argv = sys.argv[:2]
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "app.py"), encoding="utf-8").read().replace('if __name__ == "__main__":\n    main()', "")
ns = {"__file__": os.path.join(here, "app.py"), "__name__": "prowrite_app"}
exec(compile(src, "app.py", "exec"), ns)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
data_dir = os.environ.get("PROWRITE_DATA_DIR", os.path.join(here, "data"))
rows = list(csv.DictReader(open(os.path.join(data_dir, "validation.csv"), encoding="utf-8")))[:n]
model = ns["best_model_name"]()
scores = []
for r in rows:
    try:
        out = ns["improve_text"](r["source_text"], model, "greedy")
    except ns["AppError"]:          # e.g. empty output: counts as a failed, low-confidence case
        scores.append((float("-inf"), 0.0)); continue
    scores.append((out["mean_logp"], sacrebleu.sentence_chrf(out["text"], [r["target_text"]]).score))

def summary(lo, hi):
    good = [s for s in scores if s[1] >= 80]; bad = [s for s in scores if s[1] < 50]
    return {"high_threshold": hi, "low_threshold": lo,
            "share_high": sum(s[0] >= hi for s in scores) / len(scores),
            "share_low": sum(s[0] < lo for s in scores) / len(scores),
            "bad_caught_as_low": sum(s[0] < lo for s in bad) / max(1, len(bad)),
            "good_flagged_low": sum(s[0] < lo for s in good) / max(1, len(good)),
            "precision_of_high(chrF>=80)": sum(1 for s in scores if s[0] >= hi and s[1] >= 80) / max(1, sum(s[0] >= hi for s in scores)),
            "n_good": len(good), "n_bad": len(bad)}

best = None
for lo in [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.45, -0.4, -0.35, -0.3]:
    for hi in [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05]:
        if hi <= lo:
            continue
        s_ = summary(lo, hi)
        # keep good outputs almost never flagged low (≤3 %), maximise bad outputs caught; high must be ≥85 % precise
        if s_["good_flagged_low"] <= 0.03 and s_["precision_of_high(chrF>=80)"] >= 0.85:
            key = (s_["bad_caught_as_low"], s_["share_high"])
            if best is None or key > best[0]:
                best = (key, s_)
print(json.dumps({"model": model, "n_validation_messages": len(scores), **(best[1] if best else summary(-0.6, -0.25))}, indent=1))
