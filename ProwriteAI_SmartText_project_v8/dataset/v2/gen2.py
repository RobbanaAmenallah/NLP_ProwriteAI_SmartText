"""ProwriteAI dataset v2 — minimal-edit smart text improvement pairs.

Clean, house-style targets are produced from sentence frames (frames.txt) and from the
v1 professional sentence families (only informal variants that stay close to the target).
Informal sources come from hand-written informal templates of the same frame plus a
realistic noise layer. Frames/families are assigned to train/validation/test BEFORE
any variation is generated.
"""
import csv, glob, json, os, random, re, sys
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import gen as g1                       # v1 helpers: entities, misspellings, checks
from gen import (parse_families, sample_entities, MISSPELL, random_typo, target_format_ok, spell_bad,
                 has_neg, nums_ok, FIRST, DAYS, MONTHS, CITIES, ITEMS, AMOUNTS, PROJECTS)

SEED = 20261107 + 2
rnd = random.Random(SEED)
g1.rnd = rnd
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
TARGETS = json.loads(os.environ["TARGETS"]) if "TARGETS" in os.environ else {"train": 24000, "validation": 3000, "test": 3000}

# ------------------------------------------------------------------ frames
def parse_frames(path):
    frames, cat = [], None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("## cat:"):
            cat = line.split(":", 1)[1].strip(); continue
        if not line or line.startswith("#"):
            continue
        fid, rest = [x.strip() for x in line.split("|", 1)]
        parts = [x.strip() for x in rest.split("||")]
        frames.append({"fid": fid, "cat": cat, "formal": parts[0], "informals": parts[1:], "kind": "frame"})
    return frames

# ------------------------------------------------------------------ slot lexicons: (formal, [informal options])
def opts(*pairs):
    return [(p, [p.lower()]) if isinstance(p, str) else (p[0], p[1]) for p in pairs]

ACTIONS = [  # base, past, pp, ing, s3, objects
    ("send", "sent", "sent", "sending", "sends", ["the report", "the invoice", "the photos", "the slides", "the link", "the contract", "my CV", "the file", "the schedule", "the address"]),
    ("call", "called", "called", "calling", "calls", ["{name}", "the doctor", "the client", "the bank", "the landlord", "the plumber", "the restaurant", "your mom"]),
    ("finish", "finished", "finished", "finishing", "finishes", ["the project", "the report", "my homework", "the presentation", "the assignment", "the book", "the laundry", "the essay"]),
    ("book", "booked", "booked", "booking", "books", ["a table", "the meeting room", "the tickets", "a taxi", "the hotel", "a doctor's appointment", "the flights"]),
    ("fix", "fixed", "fixed", "fixing", "fixes", ["the printer", "the bug", "the sink", "my bike", "the website", "the heating", "the car", "the door"]),
    ("buy", "bought", "bought", "buying", "buys", ["the tickets", "some milk", "a gift for {name}", "new shoes", "the groceries", "a new phone", "coffee", "bread"]),
    ("pick up", "picked up", "picked up", "picking up", "picks up", ["the kids", "{name}", "the package", "the keys", "the cake", "my sister", "the documents"]),
    ("check", "checked", "checked", "checking", "checks", ["my email", "the schedule", "the numbers", "the weather", "the tracking page", "the calendar", "the menu"]),
    ("bring", "brought", "brought", "bringing", "brings", ["the charger", "some snacks", "your laptop", "the documents", "an umbrella", "the keys", "a cake"]),
    ("meet", "met", "met", "meeting", "meets", ["{name}", "the new manager", "the client", "my friends", "the team", "my cousin"]),
    ("watch", "watched", "watched", "watching", "watches", ["the game", "the movie", "the news", "the presentation", "the kids", "the new series"]),
    ("clean", "cleaned", "cleaned", "cleaning", "cleans", ["the kitchen", "my room", "the car", "the office", "the bathroom", "the fridge"]),
    ("pay", "paid", "paid", "paying", "pays", ["the rent", "the bill", "the invoice", "the electricity bill", "the parking ticket", "the deposit"]),
    ("read", "read", "read", "reading", "reads", ["your message", "the contract", "the article", "the instructions", "the book", "the email"]),
    ("write", "wrote", "written", "writing", "writes", ["the report", "the email", "the essay", "a review", "the summary", "the speech"]),
    ("cancel", "canceled", "canceled", "canceling", "cancels", ["the meeting", "the order", "the reservation", "my subscription", "the trip", "the appointment"]),
    ("visit", "visited", "visited", "visiting", "visits", ["my parents", "the museum", "{name}", "the new office", "my grandmother", "the client"]),
    ("take", "took", "taken", "taking", "takes", ["the bus", "the train", "a taxi", "a day off", "the exam", "some photos", "the dog for a walk"]),
    ("find", "found", "found", "finding", "finds", ["my keys", "the file", "a new apartment", "the charger", "a good restaurant", "my wallet"]),
    ("print", "printed", "printed", "printing", "prints", ["the tickets", "the report", "the slides", "the contract", "the boarding pass"]),
    ("order", "ordered", "ordered", "ordering", "orders", ["pizza", "the supplies", "a new laptop", "the cake", "some books", "a taxi"]),
    ("help", "helped", "helped", "helping", "helps", ["{name}", "my brother", "the new intern", "my neighbor", "the team"]),
    ("invite", "invited", "invited", "inviting", "invites", ["{name}", "the whole team", "my cousins", "everyone", "our neighbors"]),
    ("move", "moved", "moved", "moving", "moves", ["the meeting", "the boxes", "the car", "the appointment", "the table"]),
    ("prepare", "prepared", "prepared", "preparing", "prepares", ["the slides", "dinner", "the budget", "the room", "the contract", "the presentation"]),
    ("return", "returned", "returned", "returning", "returns", ["the book", "the keys", "the rental car", "the jacket", "your call"]),
    ("update", "updated", "updated", "updating", "updates", ["the spreadsheet", "the website", "my phone", "the calendar", "the client list"]),
    ("share", "shared", "shared", "sharing", "shares", ["the photos", "the document", "the slides", "the link", "my notes"]),
    ("sign", "signed", "signed", "signing", "signs", ["the contract", "the form", "the lease", "the card for {name}"]),
    ("feed", "fed", "fed", "feeding", "feeds", ["the cat", "the dog", "the fish", "the baby"]),
]
SUBJ = opts(("I", ["i", "I"]), "We", "They", "She", "He", "{name}", "My brother", "My sister", "Our team", "My manager", "My roommate")
SUBJ3 = opts("She", "He", "{name}", "My brother", "My manager", "My roommate")
S_BE = [("I am", ["i am", "im", "i'm", "i"]), ("You are", ["you are", "u are", "your", "you"]), ("We are", ["we are", "we're", "were", "we"]),
        ("They are", ["they are", "theyre", "they r", "they"]), ("She is", ["she is", "shes", "she"]), ("He is", ["he is", "hes", "he"]),
        ("{name} is", ["{name} is", "{name}s", "{name}"])]
S_WAS = [("I was", ["i was"]), ("We were", ["we were", "we was"]), ("They were", ["they were", "they was"]), ("She was", ["she was", "she were"]),
         ("He was", ["he was"]), ("{name} was", ["{name} was", "{name} were"])]
S_HAVE = [("I have", ["i have", "ive", "i've", "i has"]), ("We have", ["we have", "weve", "we has"]), ("She has", ["she has", "she have", "shes"]),
          ("He has", ["he has", "he have"]), ("{name} has", ["{name} has", "{name} have"]), ("They have", ["they have", "theyve"])]
TPAST = opts(("yesterday", ["yesterday", "yday", "yesterday", "yestarday"]), ("last night", ["last night", "lastnight", "last nite"]),
             "this morning", "last week", "on {day}", "two days ago", "earlier today", "last weekend", "last month", "on {date}")
TFUT = opts(("tomorrow", ["tomorrow", "tmrw", "2moro", "tomorow"]), ("tonight", ["tonight", "tonite"]), "next week", "on {day}",
            "this weekend", "later today", ("tomorrow morning", ["tomorrow morning", "tmrw morning"]), "next month", "on {date}", "at {time}")
TNOW = opts("today", "right now", "this week")
PLACE = opts("at the office", "at home", "at the gym", "at the library", "at the station", "at the airport", "downtown",
             "at the hospital", "in the kitchen", "at the coffee shop", "at the park", "at the mall")
PLACE2 = opts("the gym", "the beach", "the cinema", "the mall", "the park", "the library", "the doctor", "the airport",
              "my parents' house", "the supermarket", "the dentist", "the museum", "the concert")
PLACE3 = opts("the office", "the gym", "home", "the restaurant", "my friend's house", "the front desk", "the coffee shop", "school")
EVENT = opts("the meeting", "the party", "the game", "the dinner", "the concert", "the wedding", "the workshop", "the interview",
             "the class", "the training", "the barbecue", "the conference", "the team lunch", "the birthday party")
THING = opts("weekend", "trip", "exam", "interview", "first day", "vacation", "flight", "presentation", "birthday", "date", "holiday")
THING2 = opts("the instructions", "this email", "the new rules", "what you mean", "the homework", "the contract", "the question")
STATE = opts("busy", "at work", "in a meeting", "driving", "at the gym", "in class", "on the train", "with a client", "at the doctor")
FEELING = opts("excited", "nervous", "worried", "happy", "curious", "stressed")
REASON = opts(("I am sick", ["i am sick", "im sick", "i am sick"]), "my car broke down", "I missed the bus", "I have a doctor's appointment",
              "my phone died", "there was a lot of traffic", "I have to work late", "my flight was delayed", "I overslept",
              "my son is sick", "I have a family emergency", "the trains are not running")
REASON2 = opts("the traffic was terrible", "the train was delayed", "I missed the bus", "I overslept", "my alarm did not go off",
               "I could not find a parking spot", "the meeting ran late")
FOOD = opts("pizza", "sushi", "Thai food", "pasta", "tacos", "Indian food", "your cooking", "chocolate cake", "coffee")
MEAL = opts("lunch", "dinner", "breakfast", "brunch", "coffee", "drinks")
BROKEN = opts("The printer", "The Wi-Fi", "My laptop", "The elevator", "The coffee machine", "The heating", "My phone", "The projector")
BROKEN2 = opts("the printer", "my order", "the website", "the heating", "my account", "the invoice", "the Wi-Fi", "the app")
BROKEN3 = opts("heating", "shower", "washing machine", "fridge", "window", "door lock", "oven", "toilet")
WEATHER = opts("hot", "cold", "windy", "rainy")
ACTIVITY = opts("go running", "go to the beach", "play tennis", "walk to work", "have a picnic", "ride a bike")
ACTIVITY2 = opts("visit our grandparents", "go camping", "move to a new apartment", "paint the living room", "throw a party",
                 "go to the mountains", "have a barbecue", "take a short trip")
DEADLINE = opts("the meeting", "Friday", "noon", "the end of the week", "{time}", "the weekend", "{day}")
PROJECT_THING = opts("the marketing plan", "the website redesign", "the final report", "our group project", "the {project} project",
                     "the budget proposal", "the new app", "the client presentation")
COURSE = opts("math", "history", "biology", "economics", "statistics", "chemistry", "English", "physics")
GIFT = opts("the flowers", "the cake", "the book", "the present", "the card", "the scarf")
TOPIC = opts("the new schedule", "the plan", "the budget", "the trip", "the deadline", "the idea")
ACHIEVEMENT = opts("new job", "graduation", "promotion", "engagement", "new baby", "new apartment", "exam results")
NUMS = [str(n) for n in range(2, 46)]
SLOTS = {"S": SUBJ, "S3": SUBJ3, "S_be": S_BE, "S_was": S_WAS, "S_have": S_HAVE, "tpast": TPAST, "tfut": TFUT, "tnow": TNOW,
         "place": PLACE, "place2": PLACE2, "place3": PLACE3, "event": EVENT, "thing": THING, "thing2": THING2, "state": STATE,
         "feeling": FEELING, "reason": REASON, "reason2": REASON2, "food": FOOD, "meal": MEAL, "broken": BROKEN,
         "broken2": BROKEN2, "broken3": BROKEN3, "weather": WEATHER, "activity": ACTIVITY, "activity2": ACTIVITY2,
         "deadline": DEADLINE, "project_thing": PROJECT_THING, "course": COURSE, "gift": GIFT, "topic": TOPIC,
         "achievement": ACHIEVEMENT}
PH = re.compile(r"\{([A-Za-z0-9_]+)\}")
PERSON_VERBS = {"call", "meet", "help", "invite", "visit", "feed", "pick up", "watch"}
PERSON_OBJECTS = {"the kids", "my sister", "my friends", "the team", "my cousin", "the client", "the new manager", "my parents",
                  "my grandmother", "my brother", "the new intern", "my neighbor", "everyone", "our neighbors", "my cousins"}
PROPER = set(FIRST) | set(DAYS) | set(MONTHS) | set(CITIES) | set(PROJECTS) | {"I", "Thai", "Indian", "English", "Wi-Fi", "Professor", "Dr.", "Room", "CV"}

def entity_values():
    E = sample_entities({"name", "person", "person2", "prof", "day", "day2", "time", "time2", "date", "date2", "month", "min",
                         "num", "num2", "amount", "order", "ticket", "room", "project", "course", "pct", "doc", "draft",
                         "folder", "city", "role", "field", "item", "app", "mtime", "ptime", "etime", "ltime", "day3"})
    n = rnd.choice(NUMS); E["num"] = (n, n)
    return E

def fill_pair(formal_t, informal_t, E):
    """Fill one sentence template pair. Composite slots are expanded recursively with shared entities."""
    has_verb = any(k in formal_t for k in ("{base}", "{past}", "{pp}", "{ing}", "{s3}"))
    pool = ACTIONS if has_verb else [a for a in ACTIONS if a[0] not in PERSON_VERBS]
    act = rnd.choice(pool)
    objs = act[5] if has_verb else [o for o in act[5] if "{name}" not in o and o not in PERSON_OBJECTS]
    obj = rnd.choice(objs)
    choice = {}
    def val(key, side):
        if key in ("base", "past", "pp", "ing", "s3"):
            return act[("base", "past", "pp", "ing", "s3").index(key)]
        if key == "obj":
            return obj
        if key in SLOTS:
            if key not in choice:
                choice[key] = rnd.choice(SLOTS[key])
            f, infs = choice[key]
            return f if side == 0 else rnd.choice(infs)
        if key in E:
            return E[key][side]
        raise KeyError(key)
    def render(t, side):
        for _ in range(4):                      # nested slots (e.g. "on {day}")
            t2 = PH.sub(lambda m: val(m.group(1), side), t)
            if t2 == t:
                break
            t = t2
        return t
    formal = render(formal_t, 0).replace("a.m..", "a.m.").replace("p.m..", "p.m.")
    return formal, render(informal_t, 1), (act[0], obj, tuple(sorted((k, v[0]) for k, v in choice.items())))

# ------------------------------------------------------------------ v1 families → close-paraphrase sentence frames
def words(s):
    return set(re.findall(r"[a-z]+", s.lower()))

def v1_frames():
    fams = parse_families(sorted(glob.glob(os.path.join(BASE, "families", "*.txt"))))
    split_v1 = json.load(open(os.path.join(BASE, "v1_split_report.json")))["split_of"]
    out = []
    for f in fams:
        items = []
        for _, options in f.slots:
            for o in options:
                keep = [i for i in o.informals if i not in o.neg_exempt and len(words(o.formal) & words(i)) / max(1, len(words(o.formal) | words(i))) >= 0.55]
                if keep:
                    items.append((o.formal, keep))
        if items:
            out.append({"fid": "v1_" + f.fid, "cat": "v1_" + f.cat, "items": items, "kind": "v1", "split": split_v1[f.fid]})
    return out

# ------------------------------------------------------------------ greetings / closings (deterministic informal → formal)
OPEN = [("Hi {name},", ["hi {name}", "hey {name}", "heyy {name}", "yo {name}", "hiya {name}", "hii {name}"]),
        ("Hello {name},", ["hello {name}", "helo {name}"]), ("Hi,", ["hi", "hey", "heyy", "hiya", "hey boss", "hi boss"]),
        ("Hello,", ["hello", "hellooo"]), ("Good morning, {name}.", ["good morning {name}", "morning {name}", "gm {name}"]),
        ("Hi everyone,", ["hi everyone", "hey guys", "hey all", "hi all", "hey everyone"])]
CLOSE = [("Thanks.", ["thanks", "thx", "tnx", "thanks!!"]), ("Thank you.", ["thank you", "thank u", "ty"]),
         ("See you soon.", ["see you soon", "see u soon", "cya soon", "see ya soon"]), ("Talk to you later.", ["ttyl", "talk to you later", "talk later"]),
         ("Let me know.", ["lmk", "let me know"]), ("Have a good day.", ["have a good day", "have a gd day"]), ("Take care.", ["take care", "tc"])]

# ------------------------------------------------------------------ noise layer
CONTRACT = [(r"\bi am\b", ["im", "i'm", "im"]), (r"\bdo not\b", ["dont", "don't"]), (r"\bdoes not\b", ["doesnt", "doesn't"]),
            (r"\bdid not\b", ["didnt", "didn't"]), (r"\bcannot\b|\bcan not\b", ["cant", "can't"]), (r"\bwill not\b", ["wont", "won't"]),
            (r"\bis not\b", ["isnt", "isn't"]), (r"\bare not\b", ["arent", "aren't"]), (r"\bwas not\b", ["wasnt", "wasn't"]),
            (r"\bhave not\b", ["havent", "haven't"]), (r"\bhas not\b", ["hasnt", "hasn't"]), (r"\bcould not\b", ["couldnt", "couldn't"]),
            (r"\bit is\b", ["its", "it's"]), (r"\bi will\b", ["ill", "i'll"]), (r"\byou are\b", ["youre", "you're", "ur"]),
            (r"\bthey are\b", ["theyre", "they're"]), (r"\bthat is\b", ["thats", "that's"]), (r"\blet us\b", ["lets", "let's"]),
            (r"\bi would\b", ["id", "i'd"]), (r"\bwe are\b", ["we're"])]
ABBR = [(r"\byou\b", ["u"]), (r"\byour\b", ["ur"]), (r"\bare\b", ["r"]), (r"\bplease\b", ["pls", "plz"]), (r"\bthanks\b", ["thx"]),
        (r"\btomorrow\b", ["tmrw", "2moro"]), (r"\bbecause\b", ["bc", "cuz", "coz"]), (r"\bokay\b", ["ok", "k"]), (r"\bright now\b", ["rn"]),
        (r"\bgoing to\b", ["gonna"]), (r"\bwant to\b", ["wanna"]), (r"\bgot to\b", ["gotta"]), (r"\bpeople\b", ["ppl"]),
        (r"\bmessage\b", ["msg"]), (r"\bbefore\b", ["b4"]), (r"\btonight\b", ["tonite"]), (r"\breally\b", ["rly", "realy"]),
        (r"\bwith\b", ["w/", "w"]), (r"\bminutes\b", ["mins"]), (r"\bbirthday\b", ["bday"]), (r"\bsee\b", ["c"]), (r"\bto be honest\b", ["tbh"]),
        (r"\bby the way\b", ["btw"]), (r"\blet me know\b", ["lmk"]), (r"\bsomething\b", ["smth"]), (r"\bprobably\b", ["prob", "prolly"])]
CONFUSE = [(r"\bwhere\b", "were"), (r"\byour\b", "you're"), (r"\btheir\b", "there"), (r"\bthere\b", "their"), (r"\btoo\b", "to"),
           (r"\bthan\b", "then"), (r"\bknow\b", "no"), (r"\bhere\b", "hear"), (r"\bthen\b", "than"), (r"\blose\b", "loose")]
PREP = [(r"\bon (monday|tuesday|wednesday|thursday|friday)", r"in \1"), (r"\bat (\d)", r"on \1"), (r"\bin the kitchen\b", "at the kitchen")]

def sub_once(pattern, repl_choices, s, p_each):
    def rep(m):
        return rnd.choice(repl_choices) if rnd.random() < p_each else m.group(0)
    return re.sub(pattern, rep, s)

def noise(s, level):
    """level in {'low','mid','high'} — controls how many error types are applied."""
    p = {"low": 0.35, "mid": 0.6, "high": 0.85}[level]
    s = s.lower() if rnd.random() < 0.9 else s
    if rnd.random() < p:
        for pat, reps in CONTRACT:
            s = sub_once(pat, reps, s, 0.75)
    if rnd.random() < p:
        for pat, reps in ABBR:
            s = sub_once(pat, reps, s, 0.55)
    if rnd.random() < p * 0.4:
        cands = [(pat, r) for pat, r in CONFUSE if re.search(pat, s)]
        if cands:
            pat, r = rnd.choice(cands); s = re.sub(pat, r, s, count=1)
    if rnd.random() < p * 0.5:
        s = re.sub(r"[a-z']+", lambda m: rnd.choice(MISSPELL[m.group(0)]) if m.group(0) in MISSPELL and rnd.random() < 0.35 else m.group(0), s)
    if rnd.random() < p * 0.25:
        segs = list(re.finditer(r"[a-z]{6,}", s))
        if segs:
            m = rnd.choice(segs); s = s[:m.start()] + random_typo(m.group(0)) + s[m.end():]
    if rnd.random() < p * 0.2:
        m = list(re.finditer(r"\b(the|a|an) ", s))
        if m:
            mm = rnd.choice(m); s = s[:mm.start()] + s[mm.end():]
    if rnd.random() < p * 0.1:
        pat, r = rnd.choice(PREP); s = re.sub(pat, r, s, count=1)
    if rnd.random() < 0.5:
        s = s.replace("'", "")
    if rnd.random() < 0.55:
        s = re.sub(r",(?=\s|$)", "", s)          # keep digit commas such as $1,200
    return s

def finish_source(s, target):
    s = re.sub(r"\s+", " ", s).strip().rstrip(" .")
    r = rnd.random()
    if s.endswith("?") and rnd.random() < 0.5:
        s = s.rstrip("?")
    elif not s.endswith(("?", "!")):
        s += "" if r < 0.6 else rnd.choice([".", "..", "!", "?" if target.endswith("?") else ".", "!!"])
    case = rnd.random()
    if case < 0.65:
        s = s.lower()
    elif case < 0.85:
        s = s[:1].upper() + s[1:]
    return s

def light(target):
    """Nearly-correct source: 1–2 small slips."""
    s = target
    ops = rnd.sample(["lowerfirst", "comma", "final", "misspell", "lower_i", "apos", "contract", "article"], k=rnd.choice([1, 1, 2]))
    for op in ops:
        if op == "lowerfirst":
            s = s[:1].lower() + s[1:]
        elif op == "comma" and "," in s:
            s = s.replace(",", "", 1)
        elif op == "final" and s[-1] in ".!":
            s = s[:-1]
        elif op == "misspell":
            s = re.sub(r"[A-Za-z']+", lambda m: rnd.choice(MISSPELL[m.group(0)]) if m.group(0) in MISSPELL and rnd.random() < 0.5 else m.group(0), s)
        elif op == "lower_i":
            s = re.sub(r"\bI\b", "i", s)
        elif op == "apos":
            s = s.replace("'", "")
        elif op == "contract":
            s = re.sub(r"\bI am\b", "I'm", s, count=1); s = re.sub(r"\bdo not\b", "don't", s, count=1); s = re.sub(r"\bcannot\b", "can't", s, count=1)
        elif op == "article":
            s = re.sub(r" the ", " ", s, count=1)
    return s

def lower_first_if_common(sent):
    first = sent.split(" ", 1)[0]
    if first.rstrip(",.") in PROPER or re.match(r"^[A-Z]{2,}|^\d", first):
        return sent
    return sent[:1].lower() + sent[1:]
