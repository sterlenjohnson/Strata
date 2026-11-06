#!/usr/bin/env python3
"""Draw docs/img/context-decode.svg and context-prefill.svg from docs/bench/context.json (no dependencies).
Usage: python3 docs/bench/plot_context.py"""
import json, math, os
here = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(here, "context.json")))["series"]
W, H, L, Rm, T, B = 820, 600, 62, 20, 40, 150
def chart(key, title, ylabel, out):
    pts = [(s, [(x, y) for x, y in zip(s["x"], s[key]) if y is not None]) for s in data]
    ymax = max(y for _, p in pts for _, y in p)
    step = 10 ** math.floor(math.log10(ymax)); yt = [step * k for k in range(0, 12)]
    ytop = next(v for v in yt if v >= ymax * 1.05)
    ticks = [step * k for k in range(0, int(ytop / step) + 1)]
    if len(ticks) > 8: ticks = ticks[::2]
    xmin, xmax = math.log2(1536), math.log2(1400000)
    px = lambda x: L + (math.log2(x) - xmin) / (xmax - xmin) * (W - L - Rm)
    py = lambda y: T + (1 - y / ytop) * (H - T - B)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{W/2}" y="22" text-anchor="middle" font-size="15" font-weight="bold" fill="#222">{title}</text>']
    for t in ticks:
        o.append(f'<line x1="{L}" x2="{W-Rm}" y1="{py(t)}" y2="{py(t)}" stroke="#e3e3e3"/><text x="{L-6}" y="{py(t)+4}" text-anchor="end" fill="#444">{t:g}</text>')
    for k in range(11, 21):
        x = 2 ** k
        if x < 1536 or x > 1400000: continue
        lab = "1M" if x >= 1048576 else f"{x//1024}k"
        o.append(f'<line x1="{px(x)}" x2="{px(x)}" y1="{T}" y2="{H-B}" stroke="#eee"/><text x="{px(x)}" y="{H-B+16}" text-anchor="middle" fill="#444">{lab}</text>')
    o.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{H-B}" stroke="#888"/><line x1="{L}" x2="{W-Rm}" y1="{H-B}" y2="{H-B}" stroke="#888"/>')
    o.append(f'<text x="{(L+W-Rm)/2}" y="{H-B+36}" text-anchor="middle" fill="#222">prompt tokens already in context (log scale)</text>')
    o.append(f'<text transform="translate(16,{(T+H-B)/2}) rotate(-90)" text-anchor="middle" fill="#222">{ylabel}</text>')
    for s, p in pts:
        c = s["color"]
        if not p: continue
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.2" points="' + " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in p) + '"/>')
        for x, y in p: o.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.6" fill="{c}"/>')
    ly = H - B + 54
    for i, s in enumerate(data):
        cx = L + (i % 2) * 380; cy = ly + (i // 2) * 20
        o.append(f'<line x1="{cx}" x2="{cx+22}" y1="{cy}" y2="{cy}" stroke="{s["color"]}" stroke-width="2.5"/><circle cx="{cx+11}" cy="{cy}" r="3.6" fill="{s["color"]}"/><text x="{cx+30}" y="{cy+4}" fill="#222">{s["name"]}</text>')
    o.append("</svg>")
    open(os.path.join(here, "..", "img", out), "w").write("\n".join(o))
chart("decode", "Decode speed over context", "decode tok/s", "context-decode.svg")
chart("prefill", "Prefill speed over context", "prefill tok/s (average over the prompt)", "context-prefill.svg")
