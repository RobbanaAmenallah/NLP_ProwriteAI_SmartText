import os, sys, json, csv, re, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen2 import *

frames = parse_frames(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames.txt"))
v1 = v1_frames()

# ---------------- split assignment BEFORE generation
srnd = random.Random(SEED + 1)
by_cat = defaultdict(list)
for f in frames:
    by_cat[f["cat"]].append(f)
split_of = {}
for cat in sorted(by_cat):
    lst = sorted(by_cat[cat], key=lambda f: f["fid"]); srnd.shuffle(lst)
    k = max(1, round(len(lst) * 0.11))
    for f in lst[:k]: split_of[f["fid"]] = "validation"
    for f in lst[k:2 * k]: split_of[f["fid"]] = "test"
    for f in lst[2 * k:]: split_of[f["fid"]] = "train"
for f in v1:
    split_of[f["fid"]] = f["split"]
pools = {s: {"frame": [f for f in frames if split_of[f["fid"]] == s], "v1": [f for f in v1 if split_of[f["fid"]] == s]} for s in TARGETS}
for s in TARGETS:
    print(s, "frames", len(pools[s]["frame"]), "v1 families", len(pools[s]["v1"]))

def pick_sentence(split):
    if rnd.random() < 0.72 or not pools[split]["v1"]:
        f = rnd.choice(pools[split]["frame"])
        return f["fid"], f["cat"], f["formal"], rnd.choice(f["informals"])
    f = rnd.choice(pools[split]["v1"])
    formal, infs = rnd.choice(f["items"])
    return f["fid"], f["cat"], formal, rnd.choice(infs)

SEPS = [" ", " ", ", ", ". ", "... ", " - ", ". "]
reject = Counter()

def make_example(split):
    E = entity_values()
    n = rnd.choices([1, 2, 3], weights=[0.45, 0.40, 0.15])[0]
    sents, fids, cats, cores = [], [], [], []
    for _ in range(n):
        fid, cat, ft, it = pick_sentence(split)
        if fid in fids:
            continue
        f_s, i_s, core = fill_pair(ft, it, E)
        sents.append((f_s, i_s)); fids.append(fid); cats.append(cat); cores.append(core)
    mode = rnd.choices(["informal", "light", "correct"], weights=[0.78, 0.15, 0.07])[0]
    greet = rnd.choice(OPEN) if rnd.random() < 0.3 else None
    close = rnd.choice(CLOSE) if rnd.random() < 0.2 else None
    if close:
        body = " ".join(f for f, _ in sents).lower()
        if ("thank" in close[0].lower() and "thank" in body) or ("let me know" in close[0].lower() and "let me know" in body) \
                or ("see you" in close[0].lower() and "see you" in body):
            close = None
    formal_parts = [f for f, _ in sents]
    if greet:
        g_f = greet[0].replace("{name}", E["name"][0])
        first = formal_parts[0]
        if g_f.endswith(","):
            first = lower_first_if_common(first)
        formal_parts = [g_f + " " + first] + formal_parts[1:]
    if close:
        formal_parts.append(close[0])
    target = " ".join(formal_parts)
    if mode == "correct":
        source = target
    elif mode == "light":
        source = light(target)
    else:
        parts = ([rnd.choice(greet[1]).replace("{name}", E["name"][1])] if greet else []) + [i for _, i in sents] + \
                ([rnd.choice(close[1])] if close else [])
        s = parts[0]
        for p in parts[1:]:
            s += (" " if s.endswith(("?", "!", ",")) else rnd.choice(SEPS)) + p
        source = finish_source(noise(s, rnd.choice(["low", "mid", "mid", "high"])), target)
    return {"source": re.sub(r"\s+", " ", source).strip(), "target": target, "mode": mode, "fids": fids, "cats": cats,
            "cores": cores, "E": E}

def validate(ex):
    s, t = ex["source"], ex["target"]
    err = target_format_ok(t)
    if err:
        return err
    known = {v[0] for v in ex["E"].values()} | PROPER
    bad = [w for w in spell_bad(t, list(known)) if w.strip(".") not in known]
    if bad:
        return "target_spelling:" + bad[0]
    for name in set(FIRST) & set(re.findall(r"[A-Z][a-z]+", t)):
        if name.lower() not in s.lower():
            return "name_missing"
    if not nums_ok(s.replace(" dollars", ""), t):
        return "number_mismatch"
    if has_neg(s) != has_neg(t):
        return "negation_mismatch"
    ws, wt = len(s.split()), len(t.split())
    if ws < 2 or not (0.5 <= wt / ws <= 3.0):
        return "length_ratio"
    if ex["mode"] != "correct" and s == t:
        return "source_equals_target"
    return None

rows, seen_src, seen_pair = {}, set(), set()
core_count, masked_tgt = Counter(), Counter()
for split, n in TARGETS.items():
    out, attempts = [], 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        ex = make_example(split)
        e = validate(ex)
        if e:
            reject[e.split(":")[0]] += 1; continue
        ns = re.sub(r"[^a-z0-9 ]", "", ex["source"].lower())
        if (ex["source"], ex["target"]) in seen_pair:
            reject["exact_duplicate_pair"] += 1; continue
        if ns in seen_src:
            reject["duplicate_source"] += 1; continue
        mk = re.sub(r"\b(" + "|".join(FIRST + DAYS + MONTHS) + r")\b|\d[\d:.,]*", "X", ex["target"])
        if masked_tgt[mk] >= 2:
            reject["near_dup_masked_target"] += 1; continue
        ck = (tuple(ex["fids"]), tuple(c[:2] for c in ex["cores"]))
        if core_count[ck] >= 3:
            reject["near_dup_core_combination"] += 1; continue
        seen_pair.add((ex["source"], ex["target"])); seen_src.add(ns); masked_tgt[mk] += 1; core_count[ck] += 1
        out.append(ex)
    rows[split] = out
    print(split, len(out), "accepted from", attempts, "attempts")

# ---------------- leakage: content sentences shared across splits (excluding greetings/closings)
generic = {g[0].replace("{name}", "X") for g in OPEN} | {c[0] for c in CLOSE}
def sents(t):
    t = t.replace("a.m.", "am").replace("p.m.", "pm").replace("Dr.", "Dr")
    return [re.sub(r"\b(" + "|".join(FIRST) + r")\b", "X", x) for x in re.split(r"(?<=[.?!]) (?=[A-Z0-9])", t)]
sent_split = defaultdict(set)
for split, rs in rows.items():
    for r in rs:
        for x in sents(r["target"]):
            if x not in generic and len(x.split()) > 3:
                sent_split[x].add(split)
leaks = [x for x, sp in sent_split.items() if len(sp) > 1]
print("content sentences shared across splits:", len(leaks), leaks[:5])

os.makedirs(OUT, exist_ok=True)
COLS = ["unique_id", "task", "source_text", "target_text", "source_dataset", "source_license", "generation_method", "quality_status", "split"]
uid, meta = 0, []
for split in ["train", "validation", "test"]:
    rs = rows[split]; rnd.shuffle(rs)
    with open(f"{OUT}/{split}.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh); w.writerow(COLS)
        for r in rs:
            uid += 1
            w.writerow([f"pwai2_{uid:06d}", "smart_text_improvement", r["source"], r["target"], "ProwriteAI_Synthetic",
                        "original_synthetic_pending_rights_review", "synthetic_generation", "synthetic_auto_checked", split])
            meta.append({"uid": f"pwai2_{uid:06d}", "split": split, "mode": r["mode"], "frames": r["fids"], "cats": r["cats"]})
json.dump({"reject": reject, "split_of": split_of, "leaks": len(leaks)}, open(f"{OUT}/_report.json", "w"), indent=1)
json.dump(meta, open(f"{OUT}/_meta.json", "w"))
print("rejections:", dict(reject))
