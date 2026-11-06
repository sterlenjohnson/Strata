#!/usr/bin/env python3
"""Rewrite the decode/prefill tables in FORK.md (between the <!-- decode --> and <!-- prefill --> markers) from context.json."""
import json, os, re
here = os.path.dirname(os.path.abspath(__file__)); fork = os.path.join(here, "..", "..", "FORK.md")
D = json.load(open(os.path.join(here, "context.json")))["series"]
xs = [2048, 8192, 16384, 32768, 65536, 131000, 262000]
def near(s, x, key):
    c = [(abs(xx - x) / x, v) for xx, v in zip(s["x"], s[key]) if abs(xx - x) / x < 0.2 and v is not None]
    return min(c)[1] if c else None
def tab(key, fmt):
    r = ["| Model | " + " | ".join(f"{round(x/1024)}k" for x in xs) + " |", "|---|" + "---:|" * len(xs)]
    for s in D:
        r.append(f"| {s.get('short', s['name'])} | " + " | ".join("-" if near(s, x, key) is None else fmt(near(s, x, key)) for x in xs) + " |")
    return "\n".join(r)
t = open(fork).read()
for tag, key, fmt in (("decode", "decode", lambda v: f"{v:.1f}"), ("prefill", "prefill", lambda v: f"{v:,.0f}")):
    t = re.sub(rf"(<!-- {tag} -->\n).*?(\n<!-- /{tag} -->)", lambda m: m.group(1) + tab(key, fmt) + m.group(2), t, flags=re.S)
open(fork, "w").write(t)
