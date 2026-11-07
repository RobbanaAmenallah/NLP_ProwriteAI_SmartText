"""Build the ProwriteAI dataset: split families, generate, validate, dedupe, write CSVs."""
import glob, json, os, sys
from gen import *

TARGETS = json.loads(os.environ["TARGETS"]) if "TARGETS" in os.environ else {"train": 9600, "validation": 1200, "test": 1200}
MODE_P = [("informal", 0.77), ("light", 0.16), ("correct", 0.07)]
CORE_CAP = 4          # max rows sharing the same core sentence combination
MASKED_TGT_CAP = 2    # max rows sharing the same entity-masked target
JACCARD_T = 0.85      # near-duplicate threshold (masked source AND masked target)
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"

fams = parse_families(sorted(glob.glob("families/*.txt")))
by_cat = defaultdict(list)
for f in fams:
    by_cat[f.cat].append(f)

# ---- split scenario families BEFORE generating any variation --------------
srnd = random.Random(SEED + 1)
cats = sorted(by_cat)
srnd.shuffle(cats)
n_hold = max(1, round(len(fams) * 0.105))           # families per held-out split
split_of = {}
for c in cats:
    srnd.shuffle(by_cat[c])
val_cats = cats[:n_hold]
test_cats = cats[-n_hold:]
for c in cats:
    lst = by_cat[c]
    k = 0
    if c in val_cats:
        split_of[lst[k].fid] = "validation"; k += 1
    if c in test_cats:
        split_of[lst[k].fid] = "test"; k += 1
    for f in lst[k:]:
        split_of[f.fid] = "train"

def mode_pick():
    r = rnd.random(); acc = 0
    for m, p in MODE_P:
        acc += p
        if r < acc:
            return m
    return "informal"

def trigrams(s):
    w = re.findall(r"[a-z0-9{}']+", s.lower())
    return {tuple(w[i:i + 3]) for i in range(max(1, len(w) - 2))}

def jacc(a, b):
    return len(a & b) / max(1, len(a | b))

reject = Counter()
seen_pairs, seen_src = set(), set()
core_count, masked_count = Counter(), Counter()
fam_kept = defaultdict(list)    # fid -> list of (src_trigrams, tgt_trigrams)

def try_one(f):
    mode = mode_pick()
    c = compose(f, mode)
    tgt_t, src_t = c["tgt_t"], c["src_t"]
    needed = set(PH.findall(tgt_t)) | set(PH.findall(src_t))
    E = sample_entities(needed)
    tgt = fill(tgt_t, E, 0)
    if mode == "informal":
        src = apply_case(fill(src_t, E, 1))
        src = re.sub(r"\s+", " ", src).strip()
    else:
        src = fill(src_t, E, 0)
    globals()["LAST"] = (src, tgt)
    # ---------- validation ----------
    if PH.search(tgt) or PH.search(src):
        reject["unfilled_placeholder"] += 1; return None
    err = target_format_ok(tgt)
    if err:
        reject[err] += 1; return None
    if spell_bad(tgt, [v[0] for v in E.values()] + ["Professor"]):
        reject["target_spelling"] += 1; return None
    for k, (fv, iv) in E.items():
        if fv not in tgt:
            reject["entity_missing_in_target"] += 1; return None
        sv = iv if mode == "informal" else fv
        if sv.lower().replace(" ", "") not in src.lower().replace(" ", ""):
            reject["entity_missing_in_source"] += 1; return None
    if not nums_ok(src.replace(" dollars", ""), tgt):
        reject["number_mismatch"] += 1; return None
    if has_neg(src) != has_neg(tgt):
        if not c["neg_exempt"]:
            reject["negation_mismatch"] += 1; return None
    ws, wt = len(src.split()), len(tgt.split())
    if mode != "correct" and src == tgt:
        reject["source_equals_target"] += 1; return None
    if ws < 2:
        reject["source_too_short"] += 1; return None
    if not (0.55 <= wt / ws <= 3.2):
        reject["length_ratio"] += 1; return None
    # ---------- duplicates / near-duplicates ----------
    if (src, tgt) in seen_pairs:
        reject["exact_duplicate_pair"] += 1; return None
    nsrc = re.sub(r"[^a-z0-9 ]", "", src.lower()).strip()
    if nsrc in seen_src:
        reject["duplicate_source"] += 1; return None
    if masked_count[tgt_t] >= MASKED_TGT_CAP:
        reject["near_dup_masked_target_cap"] += 1; return None
    ck = (f.fid, c["core"])
    if core_count[ck] >= CORE_CAP:
        reject["near_dup_core_combination_cap"] += 1; return None
    s3, t3 = trigrams(src_t), trigrams(tgt_t)
    for ps, pt in fam_kept[f.fid]:
        if jacc(t3, pt) >= JACCARD_T and jacc(s3, ps) >= JACCARD_T:
            reject["near_dup_jaccard"] += 1; return None
    seen_pairs.add((src, tgt)); seen_src.add(nsrc)
    masked_count[tgt_t] += 1; core_count[ck] += 1
    fam_kept[f.fid].append((s3, t3))
    return {"src": src, "tgt": tgt, "mode": mode, "fid": f.fid,
            "cat": "already_correct" if mode == "correct" else f.cat,
            "core_sents": c["core_sents"]}

rows = {}
for split, n in TARGETS.items():
    sf = [f for f in fams if split_of[f.fid] == split]
    out, fails = [], Counter()
    active = list(sf)
    while len(out) < n and active:
        for f in list(active):
            if len(out) >= n:
                break
            r = None
            for _ in range(60):
                r = try_one(f)
                if r:
                    break
            if r:
                out.append(r)
            else:
                active.remove(f)
    rows[split] = out
    print(split, "families", len(sf), "rows", len(out), "exhausted", len(sf) - len(active))

# ---- leakage checks -------------------------------------------------------
generic = {o[0] for v in OPEN.values() for o in v} | {c[0] for v in CLOSE.values() for c in v}
sent_split = defaultdict(set)
for split, rs in rows.items():
    for r in rs:
        for s_ in r["core_sents"]:
            sent_split[s_].add(split)
leak_sents = [s_ for s_, sp in sent_split.items() if len(sp) > 1]
fam_split = defaultdict(set)
for split, rs in rows.items():
    for r in rs:
        fam_split[r["fid"]].add(split)
leak_fams = [f for f, sp in fam_split.items() if len(sp) > 1]
src_sets = {k: {r["src"] for r in v} for k, v in rows.items()}
tgt_sets = {k: {r["tgt"] for r in v} for k, v in rows.items()}
xs = sum(len(src_sets[a] & src_sets[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
xt = sum(len(tgt_sets[a] & tgt_sets[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
print("leaking core sentences:", len(leak_sents), "leaking families:", len(leak_fams),
      "cross-split identical sources:", xs, "targets:", xt)

# ---- write CSVs -----------------------------------------------------------
import os
os.makedirs(OUT, exist_ok=True)
COLS = ["unique_id", "task", "source_text", "target_text", "source_dataset", "source_license",
        "generation_method", "quality_status", "split"]
uid = 0
meta = []
for split in ["train", "validation", "test"]:
    rs = rows[split]
    rnd.shuffle(rs)
    with open(f"{OUT}/{split}.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        w.writerow(COLS)
        for r in rs:
            uid += 1
            r["uid"] = f"pwai_{uid:06d}"
            w.writerow([r["uid"], "smart_text_improvement", r["src"], r["tgt"], "ProwriteAI_Synthetic",
                        "original_synthetic_pending_rights_review", "synthetic_generation",
                        "synthetic_auto_checked", split])
            meta.append({"uid": r["uid"], "split": split, "family": r["fid"], "category": r["cat"], "mode": r["mode"]})
json.dump({"reject": reject, "split_of": split_of, "leak_sents": len(leak_sents), "leak_fams": leak_fams,
           "xsrc": xs, "xtgt": xt, "families": len(fams), "attempted_total": sum(reject.values()) + uid},
          open(f"{OUT}/_report.json", "w"), indent=1)
json.dump(meta, open(f"{OUT}/_meta.json", "w"))
print("rejections:", dict(reject))
