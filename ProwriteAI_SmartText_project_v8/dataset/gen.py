"""ProwriteAI Smart Text Improvement - synthetic dataset builder.

Pipeline: parse hand-authored scenario families -> assign families to splits
(before any variation is generated) -> compose messages -> inject realistic
informal noise -> validate each pair -> remove duplicates / near-duplicates ->
write train/validation/test CSVs plus a validation report.
"""
import csv, glob, json, random, re, sys
from collections import Counter, defaultdict

SEED = 20261107
rnd = random.Random(SEED)

# ----------------------------------------------------------------------------
# 1. Parsing hand-authored families
# ----------------------------------------------------------------------------
class Option:
    def __init__(self, formal, informals):
        self.formal = formal.strip()
        infs = [s.strip() for s in informals if s.strip()]
        # a trailing "~" marks a reviewed pair whose negation is expressed lexically
        # differently (e.g. "for free" vs "at no extra cost"); exempt from the parity check
        self.neg_exempt = {s[:-1].strip() for s in infs if s.endswith("~")}
        self.informals = [s[:-1].strip() if s.endswith("~") else s for s in infs]

class Family:
    def __init__(self, fid, cat, opens, closes):
        self.fid, self.cat, self.opens, self.closes = fid, cat, opens, closes
        self.slots = []  # list of (optional:bool, [Option])

def parse_families(paths):
    fams = []
    cur = None
    for p in paths:
        for ln, raw in enumerate(open(p, encoding="utf-8"), 1):
            line = raw.rstrip("\n").strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("==="):
                parts = [x.strip() for x in line[3:].split("|")]
                fid, cat = parts[0], parts[1]
                kv = dict(x.split("=") for x in parts[2:])
                cur = Family(fid, cat, kv["open"].split(","), kv["close"].split(","))
                cur.slots.append((False, []))
                fams.append(cur)
            elif line.startswith("---"):
                cur.slots.append((line == "---?", []))
            elif line.startswith(">"):
                bits = [b.strip() for b in line[1:].split("::")]
                if len(bits) < 2:
                    sys.exit(f"{p}:{ln} option without informal variants")
                cur.slots[-1][1].append(Option(bits[0], bits[1:]))
            else:
                sys.exit(f"{p}:{ln} unparsable line: {line}")
    ids = [f.fid for f in fams]
    dup = [i for i, c in Counter(ids).items() if c > 1]
    if dup:
        sys.exit(f"duplicate family ids: {dup}")
    return fams

# ----------------------------------------------------------------------------
# 2. Generic openers / closers (shared, short, register-specific)
# ----------------------------------------------------------------------------
OPEN = {
    "person": [("Hi {name},", ["hi {name}", "hey {name}", "{name},"]),
               ("Hello {name},", ["hello {name}", "helo {name}", "hii {name}"]),
               ("Dear {name},", ["dear {name}", "hello {name}"]),
               ("Good morning, {name}.", ["good morning {name}", "morning {name}"]),
               ("Hi,", ["hi", "hey", "hiya"]),
               ("Hello,", ["hello", "hey there", "hi there"])],
    "team": [("Hi everyone,", ["hi everyone", "hey everyone", "hey all"]),
             ("Hello team,", ["hello team", "hey team", "yo team"]),
             ("Hi all,", ["hi all", "hey guys", "hi guys"]),
             ("Good morning, everyone.", ["good morning everyone", "morning all", "morning everyone"])],
    "support": [("Hello,", ["hello", "hi", "hey there"]),
                ("Good afternoon,", ["good afternoon", "afternoon"]),
                ("Hi there,", ["hi there", "hey"]),
                ("Dear Customer Support Team,", ["dear customer support", "hi support team", "hello support"])],
    "agent": [("Dear {name},", ["dear {name}", "hi {name}"]),
              ("Hello {name},", ["hello {name}", "hey {name}"]),
              ("Hi {name},", ["hi {name}", "hii {name}"])],
    "prof": [("Dear Professor {prof},", ["dear professor {prof}", "hi professor {prof}", "dear prof {prof}"]),
             ("Hello Dr. {prof},", ["hello dr {prof}", "hi dr {prof}", "hello doctor {prof}"]),
             ("Good morning, Professor {prof}.", ["good morning professor {prof}", "morning prof {prof}"])],
    "friend": [("Hi {name},", ["hey {name}", "heyy {name}", "yo {name}", "hi {name}"]),
               ("Hey {name},", ["hey {name}", "hey {name}!!", "heyyy {name}"]),
               ("Hello {name},", ["hello {name}", "hii {name}"]),
               ("Hi,", ["hi", "hey", "heyy"])],
}
CLOSE = {
    "thanks": [("Thank you.", ["thanks", "thx", "ty", "thank u"]),
               ("Thank you in advance.", ["thanks in advance", "thx in advance", "tia"]),
               ("Thanks for your help.", ["thanks for the help", "thx for helping", "thanks for ur help"]),
               ("I appreciate your help.", ["appreciate it", "really appreciate it", "appreciate ur help"]),
               ("Thank you very much.", ["thanks a lot", "thank you so much", "thanks a bunch"])],
    "sorry": [("I apologize for the inconvenience.", ["sorry for the trouble", "sorry about that", "sorry for the inconvenience"]),
              ("I apologize for the short notice.", ["sorry for the short notice", "sorry its last minute", "sorry for short notice"]),
              ("Sorry for any trouble this may cause.", ["sorry for any trouble", "sorry if this causes trouble"]),
              ("I am sorry about this.", ["so sorry", "really sorry", "sorry!!"])],
    "questions": [("Please let me know if you have any questions.", ["lmk if u have questions", "let me know if any questions", "any questions just ask"]),
                  ("Feel free to reach out if you need anything else.", ["ping me if u need anything else", "just text me if u need smth else"]),
                  ("Let me know what you think.", ["lmk what u think", "thoughts?", "tell me what you think"])],
    "reply": [("I look forward to hearing from you.", ["waiting for ur reply", "hope to hear from u soon", "waiting your answer"]),
              ("Please get back to me when you can.", ["get back to me when u can", "reply when u can", "answer me when you can"])],
    "see": [("I look forward to seeing you then.", ["see u then", "see you there", "c u then"]),
            ("See you then.", ["see ya then", "cya then"])],
    "friendly": [("See you soon.", ["see ya", "cya soon", "see u soon"]),
                 ("Talk to you later.", ["ttyl", "talk later", "talk to u later"]),
                 ("Have a great day.", ["have a good one", "have a nice day", "have a great day!!"])],
}

# ----------------------------------------------------------------------------
# 3. Entity values  (formal surface, informal surface)
# ----------------------------------------------------------------------------
FIRST = ["Sarah", "Omar", "Priya", "Daniel", "Mei", "Carlos", "Fatima", "Lucas", "Aisha", "Ben",
         "Yuki", "Elena", "Karim", "Sophie", "Ravi", "Hannah", "Tom", "Leila", "Marco", "Nadia",
         "Jonas", "Grace", "Ahmed", "Chloe", "Victor", "Amira", "Ethan", "Zara", "Samir", "Olivia",
         "Mateo", "Ines", "Kwame", "Lena", "Hassan", "Julia", "Arjun", "Noor", "Felix", "Rania",
         "Diego", "Anna", "Youssef", "Emma", "Kenji", "Sana", "Paul", "Maya", "Tariq", "Isabel"]
SURN = ["Martin", "Haddad", "Chen", "Okafor", "Rossi", "Novak", "Garcia", "Bennett", "Kim", "Dubois",
        "Patel", "Silva", "Ibrahim", "Fischer", "Nakamura", "Mensah", "Kowalski", "Ferreira", "Larsen", "Ahmadi"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_ABBR = {"Monday": ["mon"], "Tuesday": ["tue", "tues"], "Wednesday": ["wed", "weds"],
            "Thursday": ["thu", "thurs"], "Friday": ["fri"]}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
PROJECTS = ["Atlas", "Phoenix", "Orion", "Harbor", "Falcon", "Cedar", "Nimbus", "Beacon",
            "Summit", "Lumen", "Horizon", "Aurora"]
CITIES = ["Lisbon", "Toronto", "Berlin", "Dubai", "Singapore", "Chicago", "Madrid", "Nairobi",
          "Seoul", "Montreal", "Vienna", "Casablanca"]
COURSES = [("BIO", 210), ("MATH", 101), ("CS", 240), ("ECON", 202), ("CHEM", 115), ("HIST", 130),
           ("PSYC", 220), ("STAT", 310), ("ENG", 105), ("PHYS", 150)]
DOCS = ["sales report", "budget spreadsheet", "Q3 summary", "client contract", "project timeline",
        "monthly report", "inventory list", "marketing plan", "risk assessment", "training schedule"]
DRAFTS = ["proposal", "press release", "project plan", "cover letter", "grant application",
          "newsletter", "user guide", "conference abstract"]
FOLDERS = ["Marketing Assets", "Finance 2026", "Client Contracts", "HR Templates", "Design Files",
           "Project Archive", "Sales Reports", "Legal Documents"]
ROLES = ["project coordinator", "data analyst", "marketing specialist", "software engineer",
         "sales manager", "project assistant", "UX designer", "financial analyst"]
FIELDS = ["data science", "public health", "finance", "mechanical engineering",
          "international relations", "environmental science", "computer science", "education"]
ITEMS = ["wireless headset", "coffee maker", "desk lamp", "pair of running shoes", "backpack",
         "blender", "phone case", "office chair", "winter jacket", "electric kettle", "keyboard",
         "smart watch", "vacuum cleaner", "laptop stand"]
AMOUNTS = ["$45", "$80", "$120", "$150", "$240", "$300", "$450", "$600", "$850", "$1,200", "$1,500", "$2,400"]
MINS = ["10", "15", "20", "25", "30", "40", "45"]
ROOMS = ["104", "214", "3B", "305", "2A", "118", "410", "12"]
APPS = ["Slack", "Teams", "Zoom", "Jira", "Trello", "Notion", "Salesforce", "Outlook"]

def fmt_time(h, m):
    suf = "a.m." if h < 12 else "p.m."
    hh = h if h <= 12 else h - 12
    return f"{hh}:{m:02d} {suf}", hh, m, ("am" if h < 12 else "pm")

def informal_time(hh, m, ap):
    if m == 0:
        return rnd.choice([f"{hh}{ap}", f"{hh} {ap}", f"{hh}{ap}", f"{hh}:00{ap}"])
    return rnd.choice([f"{hh}:{m:02d}{ap}", f"{hh}.{m:02d}{ap}", f"{hh}:{m:02d} {ap}"])

def sample_entities(needed):
    E = {}
    names = rnd.sample(FIRST, 3)
    E["name"] = (names[0], names[0]); E["person"] = (names[1], names[1])
    E["person2"] = (names[2], names[2])
    s = rnd.choice(SURN); E["prof"] = (s, s)
    d = rnd.sample(DAYS, 3)
    for key, v in zip(["day", "day2", "day3"], d):
        inf = v.lower() if rnd.random() < 0.75 else rnd.choice(DAY_ABBR[v])
        E[key] = (v, inf)
    h1 = rnd.randint(8, 15); m1 = rnd.choice([0, 0, 30])
    h2 = h1 + rnd.randint(1, 2); m2 = rnd.choice([0, 30])
    f1, a, b, c = fmt_time(h1, m1); E["time"] = (f1, informal_time(a, b, c))
    f2, a, b, c = fmt_time(h2, m2); E["time2"] = (f2, informal_time(a, b, c))
    for key, hrs in [("mtime", range(7, 12)), ("ptime", range(13, 18)), ("etime", range(18, 23)), ("ltime", [12, 12, 13])]:
        f_, a, b, c = fmt_time(rnd.choice(list(hrs)), rnd.choice([0, 0, 30]))
        E[key] = (f_, informal_time(a, b, c))
    mon = rnd.choice(MONTHS); d1 = rnd.randint(2, 16); d2 = d1 + rnd.randint(3, 12)
    def inf_date(mo, dd):
        sfx = "th" if 11 <= dd <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(dd % 10, "th")
        return rnd.choice([f"{mo.lower()} {dd}", f"{mo.lower()} {dd}{sfx}", f"{dd} {mo.lower()}", f"{mo[:3].lower()} {dd}"])
    E["date"] = (f"{mon} {d1}", inf_date(mon, d1)); E["date2"] = (f"{mon} {d2}", inf_date(mon, d2))
    m2_ = rnd.choice(MONTHS); E["month"] = (m2_, m2_.lower())
    v = rnd.choice(MINS); E["min"] = (v, v)
    v = str(rnd.randint(2, 9)); E["num"] = (v, v)
    v = str(rnd.randint(10, 60)); E["num2"] = (v, v)
    v = rnd.choice(AMOUNTS); E["amount"] = (v, v if rnd.random() < 0.7 else v[1:] + " dollars")
    v = str(rnd.randint(100000, 999999)); E["order"] = (v, v)
    v = str(rnd.randint(1000, 99999)); E["ticket"] = (v, v)
    v = rnd.choice(ROOMS); E["room"] = (v, v)
    v = rnd.choice(PROJECTS); E["project"] = (v, v)
    c, n = rnd.choice(COURSES); E["course"] = (f"{c} {n}", rnd.choice([f"{c.lower()} {n}", f"{c.lower()}{n}", f"{c} {n}"]))
    p = rnd.choice(range(10, 100, 5)); E["pct"] = (f"{p}%", rnd.choice([f"{p}%", f"{p} percent"]))
    for key, lst in [("doc", DOCS), ("draft", DRAFTS), ("folder", FOLDERS), ("city", CITIES),
                     ("role", ROLES), ("field", FIELDS), ("item", ITEMS), ("app", APPS)]:
        v = rnd.choice(lst); E[key] = (v, v)
    return {k: v for k, v in E.items() if k in needed}

PH = re.compile(r"\{([a-z0-9]+)\}")
ALL_KEYS = {"name", "person", "person2", "prof", "day", "day2", "day3", "time", "time2", "mtime", "ptime", "etime", "ltime", "date", "date2",
            "month", "min", "num", "num2", "amount", "order", "ticket", "room", "project", "course",
            "pct", "doc", "draft", "folder", "city", "role", "field", "item", "app"}

def fill(t, E, idx):
    s = PH.sub(lambda m: E[m.group(1)][idx], t)
    return s.replace("a.m..", "a.m.").replace("p.m..", "p.m.")

# ----------------------------------------------------------------------------
# 4. Noise model for informal sources
# ----------------------------------------------------------------------------
PHRASE_ABBR = [("let me know", ["lmk"]), ("as soon as possible", ["asap", "ASAP"]),
               ("by the way", ["btw"]), ("right now", ["rn", "right now"]), ("going to", ["gonna"]),
               ("want to", ["wanna"]), ("i dont know", ["idk", "dunno"]), ("i don't know", ["idk", "dunno"]),
               ("thank you", ["thank u", "ty"]), ("to be honest", ["tbh"]), ("for your information", ["fyi"]),
               ("got to", ["gotta"]), ("kind of", ["kinda"]), ("out of office", ["ooo"])]
WORD_ABBR = {"you": ["u"], "your": ["ur"], "you're": ["ur", "youre"], "please": ["pls", "plz"],
             "thanks": ["thx", "tnx"], "tomorrow": ["tmrw", "tmr", "2moro"], "tonight": ["tonite"],
             "because": ["bc", "cuz", "coz"], "okay": ["ok", "k"], "people": ["ppl"], "message": ["msg"],
             "information": ["info"], "before": ["b4"], "about": ["abt"], "probably": ["prob", "prolly"],
             "really": ["rly", "realy"], "something": ["smth", "sth"], "with": ["w/"], "without": ["w/o"],
             "though": ["tho"], "through": ["thru"], "minutes": ["mins"], "documents": ["docs"],
             "appointment": ["appt"], "are": ["r"], "and": ["&", "n"], "great": ["gr8"], "later": ["l8r"],
             "what": ["wat"], "favorite": ["fav"], "should": ["shud"], "would": ["wud"], "know": ["kno"],
             "see": ["c"], "why": ["y"], "question": ["q"], "questions": ["qs"], "picture": ["pic"],
             "pictures": ["pics"], "everyone": ["every1"], "anyone": ["any1"], "someone": ["some1"]}
RARE_ABBR = {"are", "and", "great", "later", "what", "should", "would", "know", "see", "why",
             "question", "everyone", "anyone", "someone"}
MISSPELL = {"receive": ["recieve"], "received": ["recieved"], "definitely": ["definately", "defintely"],
            "separate": ["seperate"], "tomorrow": ["tommorow", "tomorow", "tommorrow"], "until": ["untill"],
            "schedule": ["schedual", "shedule"], "which": ["wich"], "because": ["becuase", "beacuse"],
            "really": ["realy"], "address": ["adress"], "necessary": ["neccessary", "necesary"],
            "available": ["availble", "avaliable"], "appreciate": ["apreciate", "appriciate"],
            "beginning": ["begining"], "believe": ["beleive"], "business": ["buisness", "bussiness"],
            "calendar": ["calender"], "colleague": ["collegue", "coleague"], "colleagues": ["collegues"],
            "environment": ["enviroment"], "immediately": ["immediatly"], "maintenance": ["maintainance"],
            "accommodate": ["accomodate"], "writing": ["writting"], "meeting": ["meating", "meetting"],
            "meetings": ["meetigns"], "tonight": ["tonigt"], "finished": ["finshed"], "office": ["ofice"],
            "problem": ["problme", "probelm"], "project": ["projet", "porject"], "send": ["sned"],
            "sorry": ["sory", "soryy"], "thanks": ["thnks", "thanx"], "today": ["todya", "tday"],
            "would": ["woud"], "could": ["coud"], "already": ["allready", "alredy"], "with": ["wiht"],
            "the": ["teh"], "know": ["konw"], "weekend": ["weekned"], "week": ["weak"],
            "minutes": ["minuts"], "deadline": ["dedline", "deadlin"], "report": ["reprot", "rapport"],
            "presentation": ["presentaion", "presantation"], "information": ["infomation"],
            "address": ["adres"], "assignment": ["assigment", "asignment"], "professor": ["proffesor", "profesor"],
            "excited": ["exited"], "possible": ["possibel"], "question": ["qestion", "quesiton"],
            "something": ["somthing"], "different": ["diffrent"], "interested": ["intrested"],
            "experience": ["experiance"], "probably": ["probly"], "unfortunately": ["unfortunatly", "unfortunetly"],
            "approval": ["aproval"], "access": ["acess"], "access.": ["acess."], "account": ["acount"],
            "refund": ["refound"], "delivery": ["delivry"], "package": ["pakage"], "customer": ["costumer"],
            "submit": ["sumbit"], "update": ["upate"], "updated": ["upated"], "tired": ["tierd"],
            "their": ["there"], "there": ["their"], "your": ["you're"], "its": ["it's"], "then": ["than"],
            "than": ["then"], "lose": ["loose"], "quite": ["quiet"], "advice": ["advise"],
            "everything": ["evrything"], "anything": ["anythng"], "feedback": ["feeback"],
            "basically": ["basicly"], "apartment": ["appartment"], "neighbor": ["neighbour", "nieghbor"],
            "restaurant": ["restaraunt", "resturant"], "birthday": ["bday", "birthay"], "friday": ["firday"],
            "wednesday": ["wensday", "wednsday"], "february": ["febuary"], "thursday": ["thrusday"]}
KEYB = dict(zip("qwertyuiopasdfghjklzxcvbnm", "wqrtyuiopoasdfghjklkzxcvbnm"[:26]))
KEYB = {"a": "s", "s": "d", "d": "f", "e": "r", "r": "t", "t": "y", "o": "p", "i": "o", "n": "m",
        "m": "n", "l": "k", "u": "y", "c": "v", "g": "h", "h": "j", "p": "o"}

def seg_map(text, fn):
    """apply fn to non-placeholder segments only"""
    parts = re.split(r"(\{[a-z0-9]+\})", text)
    return "".join(p if PH.fullmatch(p) else fn(p) for p in parts)

def word_sub(seg, table, p, rare=None):
    def rep(m):
        w = m.group(0); lw = w.lower()
        if lw in table:
            pp = p * (0.25 if rare and lw in rare else 1)
            if rnd.random() < pp:
                return rnd.choice(table[lw])
        return w
    return re.sub(r"[A-Za-z']+", rep, seg)

def random_typo(seg):
    words = [(m.start(), m.end()) for m in re.finditer(r"[a-z']{5,}", seg)
             if not NEG.fullmatch(m.group(0)) and "'" not in m.group(0)]
    if not words:
        return seg
    s, e = rnd.choice(words); w = seg[s:e]
    i = rnd.randint(1, len(w) - 2)
    op = rnd.choice(["swap", "drop", "double", "key"])
    if op == "swap":
        w2 = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    elif op == "drop":
        w2 = w[:i] + w[i + 1:]
    elif op == "double":
        w2 = w[:i] + w[i] + w[i:]
    else:
        ch = w[i]; w2 = w[:i] + KEYB.get(ch, ch) + w[i + 1:]
    return seg[:s] + w2 + seg[e:]

def noisy(template):
    t = template
    if rnd.random() < 0.55:
        def pa(seg):
            for ph, alts in PHRASE_ABBR:
                pat = r"\b" + re.escape(ph) + r"\b"
                if re.search(pat, seg) and rnd.random() < 0.6:
                    seg = re.sub(pat, rnd.choice(alts), seg, count=1)
            return word_sub(seg, WORD_ABBR, 0.55, RARE_ABBR)
        t = seg_map(t, pa)
    if rnd.random() < 0.45:
        t = seg_map(t, lambda s: word_sub(s, MISSPELL, 0.35))
    if rnd.random() < 0.25:
        segs = re.split(r"(\{[a-z0-9]+\})", t)
        cand = [i for i, s in enumerate(segs) if not PH.fullmatch(s) and re.search(r"[a-z]{5,}", s)]
        if cand:
            i = rnd.choice(cand); segs[i] = random_typo(segs[i]); t = "".join(segs)
    if rnd.random() < 0.5:
        t = t.replace("'", "")
    if rnd.random() < 0.4:
        t = seg_map(t, lambda s: s.replace(",", ""))
    if rnd.random() < 0.08:
        t = seg_map(t, lambda s: s.replace(", ", ","))
    return t

def apply_case(s):
    r = rnd.random()
    if r < 0.6:
        return s.lower()
    if r < 0.82:
        return s[:1].upper() + s[1:]
    s = re.sub(r"\bi\b", "I", s)
    s = re.sub(r"(^|[.!?] +)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s

# ----------------------------------------------------------------------------
# 5. Light-error injection for nearly-correct sources (target unchanged)
# ----------------------------------------------------------------------------
def light_errors(template):
    ops = rnd.sample(["lowerfirst", "comma", "final", "misspell", "typo", "lower_i", "article",
                      "apos", "an", "midcap", "spacing"], k=rnd.choice([1, 2, 2, 3]))
    t = template
    for op in ops:
        if op == "lowerfirst" and t[:1] != "{":
            t = t[:1].lower() + t[1:]
        elif op == "comma" and "," in t:
            idx = [m.start() for m in re.finditer(",", t)]
            i = rnd.choice(idx); t = t[:i] + t[i + 1:]
        elif op == "final" and t[-1] in ".!":
            t = t[:-1]
        elif op == "misspell":
            t = seg_map(t, lambda s: word_sub(s, MISSPELL, 0.5))
        elif op == "typo":
            segs = re.split(r"(\{[a-z0-9]+\})", t)
            cand = [i for i, s in enumerate(segs) if not PH.fullmatch(s) and re.search(r"[a-z]{5,}", s)]
            if cand:
                i = rnd.choice(cand); segs[i] = random_typo(segs[i]); t = "".join(segs)
        elif op == "lower_i":
            t = re.sub(r"\bI\b", "i", t)
        elif op == "article":
            m = list(re.finditer(r"(?<=[a-z,] )the (?=[a-z])", t))
            if m:
                mm = rnd.choice(m); t = t[:mm.start()] + t[mm.end():]
        elif op == "apos":
            t = re.sub(r"'s\b", "s", t, count=1)
        elif op == "an":
            t = re.sub(r"\ban (?=[a-z])", "a ", t, count=1)
        elif op == "midcap":
            m = list(re.finditer(r"(?<= )[a-z]{4,}(?= )", t))
            if m:
                mm = rnd.choice(m); t = t[:mm.start()] + mm.group(0).capitalize() + t[mm.end():]
        elif op == "spacing":
            t = re.sub(r", ", ",", t, count=1) if rnd.random() < 0.5 else re.sub(r"\. ", ".  ", t, count=1)
    return t

# ----------------------------------------------------------------------------
# 6. Composition
# ----------------------------------------------------------------------------
KEEP_CAP = re.compile(r"^(I|I'm|I've|I'll|I'd|\{[a-z0-9]+\}|Room|Professor|Dr\.|Q\d|Mr\.|Ms\.)\b")

def join_formal(opener, sentences):
    out = []
    if opener:
        f = opener
        first = sentences[0]
        if f.endswith(",") and not KEEP_CAP.match(first):
            first = first[0].lower() + first[1:]
        out.append(f + " " + first)
        out += sentences[1:]
    else:
        out = sentences
    return " ".join(out)

SEPS = [" ", " ", ", ", ". ", "... ", " - ", ". ", ", "]

def join_informal(parts):
    s = parts[0]
    for p in parts[1:]:
        if s.endswith("?") or s.endswith("!") or s.endswith(","):
            s += " " + p
        else:
            s += rnd.choice(SEPS) + p
    return s

def compose(fam, mode):
    """returns dict with templates (pre-fill) for formal target and source, plus keys"""
    opener = closer = None
    # length profile: short (1-2 sentences), medium, long (most slots + opener/closer)
    prof = rnd.choices(["short", "medium", "long"], weights=[0.32, 0.40, 0.28])[0]
    p_open = {"short": 0.25, "medium": 0.5, "long": 0.8}[prof]
    p_close = {"short": 0.2, "medium": 0.45, "long": 0.75}[prof]
    if fam.opens != ["none"] and rnd.random() < p_open:
        oset = rnd.choice([o for o in fam.opens if o != "none"])
        opener = rnd.choice(OPEN[oset])
    csets = [c for c in fam.closes if c != "none"]
    if csets and rnd.random() < p_close:
        closer = rnd.choice(CLOSE[rnd.choice(csets)])
    picks, core = [], []
    neg_exempt = False
    n = len(fam.slots)
    if prof == "short":
        keep = {0} if rnd.random() < 0.45 else {0, rnd.randrange(1, n)}
    elif prof == "medium":
        keep = {i for i in range(n) if not fam.slots[i][0] or rnd.random() < 0.3}
        req_tail = [i for i in keep if i > 0]
        if len(req_tail) > 1 and rnd.random() < 0.35:
            keep.discard(rnd.choice(req_tail))
    else:
        keep = {i for i in range(n) if not fam.slots[i][0] or rnd.random() < 0.8}
    for si, (opt_flag, options) in enumerate(fam.slots):
        if si not in keep:
            core.append(-1); continue
        i = rnd.randrange(len(options)); core.append(i); picks.append(options[i])
    formal_sents = [o.formal for o in picks]
    if closer:
        cl = closer[0].lower(); joined = " ".join(formal_sents).lower()
        clash = (("thank" in cl or "appreciate" in cl) and ("thank" in joined or "appreciate" in joined)) or \
                (("apolog" in cl or "sorry" in cl) and ("apolog" in joined or "sorry" in joined)) or \
                (("let me know" in cl or "reach out" in cl) and ("let me know" in joined or "contact me" in joined)) or \
                ("look forward" in cl and "look forward" in joined) or ("get back to me" in cl and "let me know" in joined)
        if clash:
            closer = None
    if closer:
        formal_sents_c = formal_sents + [closer[0]]
    else:
        formal_sents_c = formal_sents
    tgt = join_formal(opener[0] if opener else None, formal_sents_c)
    if mode == "correct":
        src = tgt
    elif mode == "light":
        src = tgt
        for _ in range(6):
            src = light_errors(tgt)
            if src != tgt:
                break
    else:
        chosen = [rnd.choice(o.informals) for o in picks]
        neg_exempt = any(ci in o.neg_exempt for ci, o in zip(chosen, picks))
        parts = ([rnd.choice(opener[1])] if opener else []) + chosen \
                + ([rnd.choice(closer[1])] if closer else [])
        raw = join_informal(parts)
        raw = noisy(raw)
        r = rnd.random()
        raw = raw.rstrip(" .,")
        if raw.endswith("?") or raw.endswith("!"):
            if rnd.random() < 0.4:
                raw = raw.rstrip("?!")
        else:
            raw += "" if r < 0.55 else rnd.choice([".", ".", "!", "..", "?" if tgt.endswith("?") else "."])
        if rnd.random() < 0.04 and fam.cat in ("everyday", "collaboration"):
            raw += " :)"
        src = raw
    return {"tgt_t": tgt, "src_t": src, "core": tuple(core),
            "core_sents": formal_sents, "opener": opener[0] if opener else None,
            "closer": closer[0] if closer else None, "neg_exempt": neg_exempt}

# ----------------------------------------------------------------------------
# 7. Validation helpers
# ----------------------------------------------------------------------------
NEG = re.compile(r"\b(not|no|never|cannot|can't|cant|won't|wont|don't|dont|didn't|didnt|isn't|isnt|"
                 r"aren't|arent|wasn't|wasnt|weren't|werent|haven't|havent|hasn't|hasnt|hadn't|hadnt|"
                 r"couldn't|couldnt|shouldn't|shouldnt|wouldn't|wouldnt|doesn't|doesnt|nobody|nothing|"
                 r"none|neither|nor|unable|unavailable|without|w/o|no one|idk|dunno|shouldnt|cannot)\b|n't\b", re.I)

def has_neg(s):
    return bool(NEG.search(s))

NUMWORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "first": "1",
            "second": "2", "third": "3", "fourth": "4", "fifth": "5"}
NUMWORD = re.compile(r"\b(" + "|".join(NUMWORDS) + r")\b", re.I)
DIGIT_ABBR = re.compile(r"\b(b4|every1|any1|some1|2moro|gr8|l8r|2nite)\b", re.I)

def numbers(s):
    s = DIGIT_ABBR.sub(" ", s)
    s = s.replace("a.m.", "").replace("p.m.", "")
    toks = re.findall(r"\d+", s)
    return sorted(t for t in toks if t != "00")

def nums_ok(a, b):
    """every digit group on one side must appear on the other side as digits or as a number word"""
    da, db = Counter(numbers(a)), Counter(numbers(b))
    wa = Counter(NUMWORDS[w.lower()] for w in NUMWORD.findall(DIGIT_ABBR.sub(" ", a)))
    wb = Counter(NUMWORDS[w.lower()] for w in NUMWORD.findall(DIGIT_ABBR.sub(" ", b)))
    return not (da - (db + wb)) and not (db - (da + wa))

from spellchecker import SpellChecker
SPELL = SpellChecker()
EXTRA_OK = {"wfh", "p.m", "a.m", "dr", "q3", "ux", "hr", "it", "vpn", "csv", "pdf", "api", "qa", "faq",
            "url", "id", "ids", "sql", "ui", "cpu", "wi-fi", "wifi", "e-mail", "online", "handover",
            "onboarding", "teammates", "teammate", "timeline", "follow-up", "parent-teacher", "headcount",
            "workaround", "checkout", "login", "dashboard", "dashboards", "screenshot", "screenshots",
            "walkthrough", "backend", "frontend", "rollout", "rollback", "username", "spreadsheet",
            "spreadsheets", "a.m.", "p.m.", "read-only", "write-up", "sign-up", "co-worker", "laptop",
            "laptops", "email", "emails", "emailed", "toolkit", "redesign", "deprioritize", "deliverables",
            "deliverable", "subfolder", "website", "app", "apps", "voicemail", "carpool", "potluck",
            "proofread", "proofreading", "reschedule", "rescheduled", "unsubscribe", "barbecue", "wednesday",
            "inbox", "onboarded", "timesheets", "timesheet", "standup", "stand-up", "mockups", "mockup", "prototype", "apa", "refactored", "refactor", "offsite"}

def spell_bad(text, entity_vals):
    t = text
    for v in sorted(entity_vals, key=len, reverse=True):
        if v:
            t = t.replace(v, " ")
    words = re.findall(r"[A-Za-z][A-Za-z'\-\.]*[A-Za-z]|[A-Za-z]", t)
    bad = []
    for w in words:
        lw = w.lower().strip(".")
        if lw in EXTRA_OK:
            continue
        base = re.sub(r"'s$", "", lw)
        parts = base.split("-")
        if all((p in SPELL) or p in EXTRA_OK or p in ("t", "s", "ll", "m", "re", "ve", "d") for p in parts if p):
            continue
        if base.replace("'", "") in SPELL:
            continue
        bad.append(w)
    return bad

def target_format_ok(t):
    if not re.match(r"^[A-Z0-9\"$#]", t):
        return "target_not_capitalized"
    if t[-1] not in ".?!":
        return "target_no_final_punct"
    if "  " in t or " ," in t or " ." in t or ",," in t or ".." in t or "{" in t or "}" in t:
        return "target_spacing_or_placeholder"
    tt = t.replace("a.m.", "am").replace("p.m.", "pm").replace("Dr.", "Dr")
    if re.search(r"[.!?] [a-z]", tt):
        return "target_sentence_case"
    if re.search(r"\b(\w+) \1\b", tt, re.I):
        return "target_repeated_word"
    if re.search(r"\bi\b", tt):
        return "target_lowercase_i"
    if re.search(r"\ba (?!u|one|UX|eu)[aeiouAEIOU]\w", tt) or re.search(r"\ban (?![aeiouAEIOUh])\w", tt):
        return "target_article_a_an"
    return None
