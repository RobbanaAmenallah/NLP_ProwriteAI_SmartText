"""Author-time checks on hand-written families: spelling of targets, negation and
number parity between each formal option and every informal variant, placeholder
consistency, and combinatorial capacity per family."""
import glob, re, sys
from gen import *

fams = parse_families(sorted(glob.glob("families/*.txt")))
problems = 0
cats = Counter()
for f in fams:
    cats[f.cat] += 1
    for o in f.opens:
        assert o in OPEN or o == "none", (f.fid, o)
    for c in f.closes:
        assert c in CLOSE or c == "none", (f.fid, c)
    combos = 1
    for optf, opts in f.slots:
        combos *= len(opts) + (1 if optf else 0)
        for o in opts:
            ph_f = set(PH.findall(o.formal))
            if ph_f - ALL_KEYS:
                print("UNKNOWN PH", f.fid, ph_f - ALL_KEYS); problems += 1
            bad = spell_bad(PH.sub("", o.formal), [])
            if bad:
                print("SPELL", f.fid, bad, "|", o.formal); problems += 1
            fmt = target_format_ok(PH.sub("X", o.formal))
            if fmt:
                print("FORMAT", fmt, f.fid, "|", o.formal); problems += 1
            for inf in o.informals:
                if set(PH.findall(inf)) != ph_f:
                    print("PH MISMATCH", f.fid, "|", o.formal, "||", inf); problems += 1
                if inf not in o.neg_exempt and has_neg(o.formal) != has_neg(inf):
                    print("NEG", f.fid, "|", o.formal, "||", inf); problems += 1
                if not nums_ok(PH.sub("", o.formal), PH.sub("", inf)):
                    print("NUM", f.fid, "|", o.formal, "||", inf); problems += 1
    if combos < 30:
        print("LOW CAPACITY", f.fid, combos); problems += 1
where = defaultdict(set)
for f in fams:
    for _, opts in f.slots:
        for o in opts:
            where[o.formal].add(f.fid)
for s_, fs in where.items():
    if len(fs) > 1:
        print("SHARED SENTENCE", fs, s_); problems += 1
print("families:", len(fams), dict(cats))
print("problems:", problems)
