"""Independent verification of the delivered CSV files (reads only the CSVs)."""
import csv, re, sys, json
from collections import Counter, defaultdict
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from gen import (target_format_ok, spell_bad, nums_ok, has_neg, OPEN, CLOSE, FIRST, SURN, PROJECTS,
                 CITIES, FOLDERS, ROLES, FIELDS, ITEMS, DOCS, DRAFTS, APPS, DAYS, MONTHS, COURSES)

D = sys.argv[1]
COLS = ["unique_id", "task", "source_text", "target_text", "source_dataset", "source_license",
        "generation_method", "quality_status", "split"]
FIXED = {"task": "smart_text_improvement", "source_dataset": "ProwriteAI_Synthetic",
         "source_license": "original_synthetic_pending_rights_review",
         "generation_method": "synthetic_generation", "quality_status": "synthetic_auto_checked"}
proper = set(FIRST + SURN + PROJECTS + CITIES + DAYS + MONTHS + APPS + ["Professor", "Dr.", "Room", "Building",
             "Thai", "Italian", "English", "French", "Excel", "Chrome", "Firefox", "HR", "IT", "QA", "API", "VPN",
             "PDF", "APA", "UX", "Q3", "Customer", "Support", "Team"]) | {c for c, _ in COURSES}
for lst in [FOLDERS, ROLES, FIELDS, ITEMS, DOCS, DRAFTS]:
    for v in lst:
        proper |= set(v.split())

issues = Counter(); rows = {}
for split in ["train", "validation", "test"]:
    raw = open(f"{D}/{split}.csv", "rb").read()
    raw.decode("utf-8")  # raises if not valid UTF-8
    rdr = csv.reader(raw.decode("utf-8").splitlines())
    header = next(rdr)
    assert header == COLS, header
    rs = []
    for r in rdr:
        if len(r) != 9:
            issues["wrong_column_count"] += 1; continue
        d = dict(zip(COLS, r))
        for c in COLS:
            if not d[c].strip():
                issues["empty_field"] += 1
        for k, v in FIXED.items():
            if d[k] != v:
                issues["bad_fixed_value_" + k] += 1
        if d["split"] != split:
            issues["split_column_mismatch"] += 1
        rs.append(d)
    rows[split] = rs

allr = [r for v in rows.values() for r in v]
ids = Counter(r["unique_id"] for r in allr)
issues["duplicate_ids"] = sum(c - 1 for c in ids.values() if c > 1)
pairs = Counter((r["source_text"], r["target_text"]) for r in allr)
issues["exact_duplicate_pairs"] = sum(c - 1 for c in pairs.values() if c > 1)
srcs = Counter(r["source_text"] for r in allr)
issues["duplicate_sources"] = sum(c - 1 for c in srcs.values() if c > 1)
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
nsrc = Counter(norm(r["source_text"]) for r in allr)
issues["duplicate_sources_after_normalization"] = sum(c - 1 for c in nsrc.values() if c > 1)
tg = Counter(r["target_text"] for r in allr)
dup_t = sum(c - 1 for c in tg.values() if c > 1)

for r in allr:
    s, t = r["source_text"], r["target_text"]
    e = target_format_ok(t)
    if e:
        issues[e] += 1
    bad = [w for w in spell_bad(t, []) if re.sub(r"'s$", "", w).strip(".") not in proper]
    if bad:
        issues["target_unknown_word"] += 1
    if not nums_ok(s.replace(" dollars", ""), t):
        issues["number_mismatch"] += 1
    if s.lower().strip() != t.lower().strip() and abs(len(t.split()) / max(1, len(s.split()))) > 3.2:
        issues["length_ratio"] += 1

# leakage: sentences shared across splits (excluding generic openers/closers)
generic = {o[0] for v in OPEN.values() for o in v} | {c[0] for v in CLOSE.values() for c in v}
def sents(t):
    t = t.replace("a.m.", "am").replace("p.m.", "pm").replace("Dr.", "Dr")
    parts = re.split(r"(?<=[.?!]) (?=[A-Z0-9\"])", t)
    out = []
    for p in parts:
        p = re.sub(r"^(Hi|Hello|Dear|Hey|Good morning|Good afternoon)[^,]{0,40}, ", "", p)
        out.append(re.sub(r"\b(" + "|".join(FIRST + SURN + PROJECTS + CITIES + DAYS + MONTHS) + r")\b", "X", p))
    return out
gen_masked = {re.sub(r"\{[a-z0-9]+\}", "X", g).replace("a.m.", "am") for g in generic}
sent_split = defaultdict(set)
for split, rs in rows.items():
    for r in rs:
        for s_ in sents(r["target_text"]):
            if s_ not in gen_masked and len(s_.split()) > 2:
                sent_split[re.sub(r"\d[\d,.:%]*", "N", s_)].add(split)
leaky = [s_ for s_, sp in sent_split.items() if len(sp) > 1]
xs = {a: {r["source_text"] for r in rows[a]} for a in rows}
xt = {a: {r["target_text"] for r in rows[a]} for a in rows}
cross_src = sum(len(xs[a] & xs[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
cross_tgt = sum(len(xt[a] & xt[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])

res = {"rows": {k: len(v) for k, v in rows.items()}, "total": len(allr), "issues": dict(issues),
       "duplicate_targets_with_different_sources": dup_t, "unique_sources": len(srcs), "unique_targets": len(tg),
       "source_equals_target_rows": sum(r["source_text"] == r["target_text"] for r in allr),
       "cross_split_identical_sources": cross_src, "cross_split_identical_targets": cross_tgt,
       "content_sentences_shared_across_splits": len(leaky), "leaky_examples": leaky[:10]}
print(json.dumps(res, indent=1))
