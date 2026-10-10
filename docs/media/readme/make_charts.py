#!/usr/bin/env python3
"""docs/media/readme/make_charts.py - the README's charts as static SVG, a light and a dark file each.

GitHub shows a README's images without scripts or page styles, so every chart is drawn here: horizontal bars with
their values written at the end (the light lime is under 3:1 on white, so the numbers carry it), a legend whenever
there are two series, and the README picks the light or dark file with <picture>. Palettes checked with the
dataviz validator (CVD separation, normal-vision floor, contrast) on GitHub's surfaces (#ffffff, #0d1117).

    python3 docs/media/readme/make_charts.py      # rewrites the *.svg beside it
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"
THEME = {
    "light": {"ink": "#1f2328", "ink2": "#59636e", "grid": "#d1d9e0", "one": "#97c425", "two": "#1b9247",
              "single": "#1b9247", "m1": "#2a78d6", "m2": "#eb6834", "r128": "#4a3aa7", "r64": "#eda100", "surface": "#ffffff",
              "hq4": "#eb6834", "hq6": "#14958f", "hq8": "#4a3aa7"},
    "dark": {"ink": "#f0f6fc", "ink2": "#9198a1", "grid": "#3d444d", "one": "#79a200", "two": "#00762c",
             "single": "#00762c", "m1": "#3987e5", "m2": "#d95926", "r128": "#9085e9", "r64": "#c98500", "surface": "#0d1117",
             "hq4": "#d95926", "hq6": "#2aa198", "hq8": "#9085e9"},
}
GPU = [("one", "RTX 3090"), ("two", "RTX 3090 + RTX 5060 Ti")]

# ---- the measurements (Ryzen 7 5700X, 128 GB DDR4-3200; decode = six-prompt geometric mean, tokens/s) ----
PROMPTS = ["Spanish chat", "Reasoning", "After an 18K document", "Code", "Edit (3.3K script)", "After a 5K prompt"]
DECODE = {  # prompt order as PROMPTS; mean of the two runs of each config with --kv-grow + STRATA_POOL_FUSED=1
    "iq4": {"one": [89.9, 97.5, 79.5, 99.1, 132.8, 70.0], "two": [107.3, 125.3, 104.4, 130.2, 181.2, 75.1]},
    "q4": {"one": [70.2, 83.7, 59.3, 74.8, 97.0, 51.7], "two": [88.1, 87.2, 78.0, 91.3, 133.0, 57.1]},
}   # the two-GPU columns: 2026-10-10, the current engine with --pcie-frac 0.25 (two runs each)
MEAN = {"iq4": {"one": 92.9, "two": 116.4}, "q4": {"one": 71.2, "two": 86.4}}
PREFILL_LABELS = ["18,076 tokens", "5,296 tokens", "3,340 tokens"]
PREFILL = {"iq4": {"one": [1803, 1180, 851], "two": [1851, 1557, 1110]},   # two GPUs: median of four runs (10-10)
           "q4": {"one": [1730, 899, 660], "two": [1774, 1115, 803]}}
STEPS = [("eddoursul custom + fixes", 77.3), ("+ IQ4_XS AVX-2 kernel", 78.0), ("+ AVX2 gather (IQ3_S)", 80.9),
         ("+ --adapt-decay 0.92", 83.3),
         ("+ upstream PRs #863, #851, #606", 83.6), ("+ elastic K/V (--kv-grow)", 90.9),
         ("+ fused CPU pool", 92.9)]
DECAY = [(0.7, 76.8), (0.85, 82.1), (0.92, 83.6), (0.95, 81.0), (0.97, 81.3), (0.99, 70.8)]
# 64 GB PC emulated on the same machine (the current engine; the 128 GB column is the README's runs): the engine (and its page cache) limited to 60 GiB with a cgroup; UD-Q4_K_XL
# (71.7 GiB of experts) then reads its experts from the NVMe through --mmap-experts.  None until measured.
RAM_CATS = ["UD-IQ4_XS · RTX 3090", "UD-IQ4_XS · 3090 + 5060 Ti", "UD-Q4_K_XL · RTX 3090", "UD-Q4_K_XL · 3090 + 5060 Ti"]
RAM64 = {"128": [92.9, 116.3, 71.2, 84.9],
         "64": [92.7, 115.7, 25.9, 39.7]}
DECODE64 = {"iq4": {"one": [89.8, 97.3, 79.0, 99.2, 132.8, 69.6], "two": [112.4, 120.2, 104.4, 127.7, 183.3, 72.8]}, "q4": {"one": [25.2, 30.5, 19.9, 28.3, 44.2, 15.7], "two": [37.3, 39.7, 32.9, 43.1, 71.3, 26.1]}}
PREFILL64 = {"iq4": {"one": [1813, 1180, 851], "two": [1833, 1550, 1084]}, "q4": {"one": [438, 162, 147], "two": [491, 213, 157]}}
# decode right after a long prompt (128 tokens of answer), UD-IQ4_XS: one GPU / two
LONG_CATS = ["after 32K tokens", "after 64K", "after 120K", "after 200K"]
LONG_DEC_Q4 = {"one": [58.6, 77.0, 48.4, 41.6], "two": [83.0, 90.0, 69.8, 74.0]}
LONG_DEC = {"one": [73.5, 93.6, 66.2, 60.0], "two": [94.7, 99.9, 98.1, 97.3]}
# huihui-ai's abliterated Flash-Next on both cards: UD-Q4_K_XL against Q8_0 with 11 + 11 GiB of experts only in VRAM
# and UD-Q6_K_XL (made here from the Q8_0) with 6 + 6 GiB (2026-10-10, the current engine; decode two runs each, three
# for UD-Q6_K_XL; prompt reading the median of four (UD-Q4_K_XL), three (UD-Q6_K_XL) and eight (Q8_0) runs; the long
# prompts one run each, two for UD-Q6_K_XL)
HQ = [("hq4", "UD-Q4_K_XL"), ("hq6", "UD-Q6_K_XL"), ("hq8", "Q8_0")]
HQ_DECODE = {"hq4": [76.5, 95.0, 77.5, 89.8, 131.0, 53.7], "hq6": [60.0, 59.5, 54.0, 54.2, 63.2, 46.4],
             "hq8": [39.6, 44.4, 43.7, 41.5, 41.4, 40.8]}
HQ_PREFILL = {"hq4": [1782, 1120, 808], "hq6": [1457, 728, 518], "hq8": [1330, 610, 426]}
HQ_LONG = {"hq4": [76.1, 92.3, 74.8, 71.0], "hq6": [51.9, 65.1, 41.7, 51.9],
           "hq8": [40.0, 46.6, 33.5, 40.4]}   # decode right after the long prompts
# KL divergence from Q8_0 (x 1000), teacher-forced over 19,208 tokens (Spanish, English docs, C++ and Python code)
KL_BIG = [("Q8_0, run again", 21.2), ("UD-Q6_K_XL", 17.7), ("Huihui UD-Q4_K_XL", 64.1),
          ("unsloth UD-Q4_K_XL (not abliterated)", 40.6)]
# KL divergence from Q8_0 (x 1000), teacher-forced over 2,304 tokens of held-out Spanish text
KL = [("Q8_0, run again", 5.6), ("Q8_0 + route-resident 0.1", 13.6), ("Q8_0 + route-resident 0.25", 23.5),
      ("Q8_0 + route-resident 0.5", 37.5), ("UD-Q4_K_XL", 28.2)]
WORKERS = {"UD-IQ4_XS": [(4, 73.8), (5, 75.7), (6, 81.8), (7, 83.6)], "UD-Q4_K_XL": [(3, 64.8), (4, 64.2), (5, 64.6), (6, 63.7)]}


def num(v, d=1):
    return f"{v:,.{d}f}"


def svg(w, h, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="{label}" font-family="{FONT}">\n<title>{label}</title>\n{body}</svg>\n')


def bar(x0, y, w, h):
    """A bar anchored at x0 whose data end is rounded (4 px)."""
    r = min(4, w / 2, h / 2)
    return f"M{x0:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} a{r},{r} 0 0 1 {-r},{r} h{-(w - r):.1f} z"


def legend(items, t, x=0, y=14):
    out, cx = [], x
    for key, text in items:
        out.append(f'<rect x="{cx}" y="{y - 10}" width="12" height="12" rx="3" fill="{t[key]}"/>')
        out.append(f'<text x="{cx + 18}" y="{y}" font-size="13" fill="{t["ink2"]}">{text}</text>')
        cx += 18 + 7.4 * len(text) + 22
    return "\n".join(out)


def ticks(vmax):
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if vmax / step <= 6:
            return step
    return 2000


def grouped(name, cats, series, label, t, vmax=None, dec=1, lab_w=180, w=760, title=None):
    """Horizontal grouped bars: one group per category, one bar per series (2 px gap inside a group); `title`, the
    chart's own name above the legend, for the charts that differ only by model."""
    bh, gap, ggap, top = 14, 2, 16, 34 + (24 if title else 0)
    n = len(series)
    gh = n * bh + (n - 1) * gap
    h = top + len(cats) * (gh + ggap) + 26
    vmax = vmax or max(max(v) for _, _, v in series)
    step = ticks(vmax)
    vmax = (int(vmax / step) + 1) * step
    pw = w - lab_w - 60
    X = lambda v: lab_w + v / vmax * pw
    body = [f'<text x="0" y="16" font-size="15" font-weight="600" fill="{t["ink"]}">{title}</text>' if title else "",
            legend([(k, s) for k, s, _ in series], t, y=38 if title else 14) if n > 1 else ""]
    for i in range(0, int(vmax) + 1, step):
        x = X(i)
        body.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{h - 22}" stroke="{t["grid"]}" stroke-width="1"/>')
        body.append(f'<text x="{x:.1f}" y="{h - 6}" font-size="11" fill="{t["ink2"]}" text-anchor="middle">{num(i, 0)}</text>')
    for ci, cat in enumerate(cats):
        y0 = top + ci * (gh + ggap)
        body.append(f'<text x="{lab_w - 10}" y="{y0 + gh / 2 + 4.5:.1f}" font-size="13" fill="{t["ink"]}" '
                    f'text-anchor="end">{cat}</text>')
        for si, (key, _, vals) in enumerate(series):
            y = y0 + si * (bh + gap)
            v = vals[ci]
            body.append(f'<path d="{bar(lab_w, y, X(v) - lab_w, bh)}" fill="{t[key]}"/>')
            body.append(f'<text x="{X(v) + 6:.1f}" y="{y + bh - 3}" font-size="12" fill="{t["ink2"]}">{num(v, dec)}</text>')
    return svg(w, h, "\n".join(body), label)


def line(name, series, label, t, xlab, xs, ylo, yhi, ystep, xfmt, w=760, h=300, mark=None, ordinal=False):
    """Lines with 2 px strokes and 8 px markers; the last point of each series labeled directly."""
    l, r, top, bot = 54, 150, 30, 40
    pw, ph = w - l - r, h - top - bot
    xlo, xhi = xs[0], xs[-1]
    X = (lambda x: l + xs.index(x) / (len(xs) - 1) * pw) if ordinal else (lambda x: l + (x - xlo) / (xhi - xlo) * pw)
    Y = lambda y: top + (yhi - y) / (yhi - ylo) * ph
    body = []
    if len(series) > 1:
        body.append(legend([(k, s) for k, s, _ in series], t, x=l))
    for y in range(ylo, yhi + 1, ystep):
        body.append(f'<line x1="{l}" y1="{Y(y):.1f}" x2="{l + pw}" y2="{Y(y):.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        body.append(f'<text x="{l - 8}" y="{Y(y) + 4:.1f}" font-size="11" fill="{t["ink2"]}" text-anchor="end">{y}</text>')
    for x in xs:
        body.append(f'<text x="{X(x):.1f}" y="{h - 20}" font-size="11" fill="{t["ink2"]}" text-anchor="middle">{xfmt(x)}</text>')
    body.append(f'<text x="{l + pw / 2:.1f}" y="{h - 4}" font-size="12" fill="{t["ink2"]}" text-anchor="middle">{xlab}</text>')
    body.append(f'<text x="12" y="{top + ph / 2:.1f}" font-size="12" fill="{t["ink2"]}" text-anchor="middle" '
                f'transform="rotate(-90 12 {top + ph / 2:.1f})">tokens/s</text>')
    for key, name_, pts in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
        body.append(f'<path d="{d}" fill="none" stroke="{t[key]}" stroke-width="2" stroke-linejoin="round"/>')
        for x, y in pts:
            body.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4.5" fill="{t[key]}" stroke="{t["surface"]}" stroke-width="2"/>')
        lx, ly = pts[-1]
        body.append(f'<text x="{X(lx) + 10:.1f}" y="{Y(ly) + 4:.1f}" font-size="12" fill="{t["ink"]}">{name_} {num(ly)}</text>')
    if mark:
        mx, my, text = mark
        body.append(f'<text x="{X(mx):.1f}" y="{Y(my) - 12:.1f}" font-size="12" fill="{t["ink"]}" text-anchor="middle">{text}</text>')
    return svg(w, h, "\n".join(body), label)


def main():
    for mode, t in THEME.items():
        charts = {
            "summary": grouped("summary", ["UD-IQ4_XS", "UD-Q4_K_XL"],
                               [(k, s, [MEAN["iq4"][k], MEAN["q4"][k]]) for k, s in GPU],
                               "Decode speed, six-prompt mean: one GPU or two", t, lab_w=120, vmax=135),
            "decode-iq4xs": grouped("d", PROMPTS, [(k, s, DECODE["iq4"][k]) for k, s in GPU],
                                    "UD-IQ4_XS decode speed per prompt", t, title="UD-IQ4_XS · decode per prompt (tokens/s)"),
            "decode-q4kxl": grouped("d", PROMPTS, [(k, s, DECODE["q4"][k]) for k, s in GPU],
                                    "UD-Q4_K_XL decode speed per prompt", t, vmax=max(DECODE["iq4"]["two"]),
                                    title="UD-Q4_K_XL · decode per prompt (tokens/s)"),
            "prefill-iq4xs": grouped("p", PREFILL_LABELS, [(k, s, PREFILL["iq4"][k]) for k, s in GPU],
                                     "UD-IQ4_XS prompt reading speed", t, dec=0, lab_w=120,
                                     title="UD-IQ4_XS · prompt reading (tokens/s)"),
            "prefill-q4kxl": grouped("p", PREFILL_LABELS, [(k, s, PREFILL["q4"][k]) for k, s in GPU],
                                     "UD-Q4_K_XL prompt reading speed", t, vmax=max(PREFILL["iq4"]["two"]), dec=0,
                                     lab_w=120, title="UD-Q4_K_XL · prompt reading (tokens/s)"),
            "steps": grouped("s", [s for s, _ in STEPS], [("single", "", [v for _, v in STEPS])],
                             "UD-IQ4_XS on the RTX 3090: what each change gave", t, lab_w=250),
            "decay": line("decay", [("single", "", DECAY)], "UD-IQ4_XS on the RTX 3090: --adapt-decay", t,
                          "--adapt-decay (the share of the routing counts kept after each cache update)",
                          [x for x, _ in DECAY], 60, 90, 10, lambda x: f"{x:.2f}", mark=(0.92, 83.6, "0.92 (used)"), ordinal=True),
            "workers": line("workers", [("m1", "UD-IQ4_XS", WORKERS["UD-IQ4_XS"]), ("m2", "UD-Q4_K_XL", WORKERS["UD-Q4_K_XL"])],
                            "Decode speed by CPU pool workers, RTX 3090 alone", t, "CPU pool workers (one per physical core)",
                            [3, 4, 5, 6, 7], 55, 90, 5, str),
        }
        lmax = max(LONG_DEC["two"] + LONG_DEC_Q4["two"])
        charts["longctx-iq4xs"] = grouped("l", LONG_CATS, [(k, n, LONG_DEC[k]) for k, n in GPU],
                                          "UD-IQ4_XS decode right after a long prompt", t, vmax=lmax, lab_w=150,
                                          title="UD-IQ4_XS · decode right after a long prompt (tokens/s)")
        charts["longctx-q4kxl"] = grouped("l", LONG_CATS, [(k, n, LONG_DEC_Q4[k]) for k, n in GPU],
                                          "UD-Q4_K_XL decode right after a long prompt", t, vmax=lmax, lab_w=150,
                                          title="UD-Q4_K_XL · decode right after a long prompt (tokens/s)")
        for m, nm in (("iq4", "UD-IQ4_XS"), ("q4", "UD-Q4_K_XL")):
            charts[f"decode-{'iq4xs' if m == 'iq4' else 'q4kxl'}-64"] = grouped(
                "d", PROMPTS, [(k, s, DECODE64[m][k]) for k, s in GPU], f"{nm} decode per prompt with 64 GB of RAM", t,
                vmax=max(DECODE["iq4"]["two"] + DECODE64["iq4"]["two"]), title=f"{nm} · decode per prompt, 64 GB of RAM (tokens/s)")
            charts[f"prefill-{'iq4xs' if m == 'iq4' else 'q4kxl'}-64"] = grouped(
                "p", PREFILL_LABELS, [(k, s, PREFILL64[m][k]) for k, s in GPU], f"{nm} prompt reading with 64 GB of RAM", t,
                vmax=max(PREFILL["iq4"]["two"]), dec=0, lab_w=120, title=f"{nm} · prompt reading, 64 GB of RAM (tokens/s)")
        charts["q8-decode"] = grouped("d", PROMPTS, [(k, s, HQ_DECODE[k]) for k, s in HQ],
                                      "Huihui decode per prompt on both cards: UD-Q4_K_XL, UD-Q6_K_XL and Q8_0", t,
                                      title="Huihui, RTX 3090 + RTX 5060 Ti · decode per prompt (tokens/s)")
        charts["q8-prefill"] = grouped("p", PREFILL_LABELS, [(k, s, HQ_PREFILL[k]) for k, s in HQ],
                                       "Huihui prompt reading on both cards: UD-Q4_K_XL, UD-Q6_K_XL and Q8_0", t, dec=0, lab_w=120,
                                       title="Huihui, RTX 3090 + RTX 5060 Ti · prompt reading (tokens/s)")
        if HQ_LONG:
            charts["q8-longctx"] = grouped("l", LONG_CATS, [(k, s, HQ_LONG[k]) for k, s in HQ],
                                           "Huihui decode right after a long prompt: UD-Q4_K_XL, UD-Q6_K_XL and Q8_0", t, lab_w=150,
                                           title="Huihui, both cards · decode right after a long prompt (tokens/s)")
        charts["q8-quality"] = grouped("k", [c for c, _ in KL], [("single", "", [v for _, v in KL])],
                                       "KL divergence from Q8_0 (x 1000): a second Q8_0 run, route-resident, UD-Q4_K_XL",
                                       t, lab_w=230, title="KL divergence from Q8_0, x 1000 (lower = closer to Q8_0)")
        charts["q6-quality"] = grouped("k", [c for c, _ in KL_BIG], [("single", "", [v for _, v in KL_BIG])],
                                       "KL divergence from Q8_0 (x 1000) over 19,208 teacher-forced tokens: a second Q8_0 run, "
                                       "UD-Q6_K_XL, Huihui's and unsloth's UD-Q4_K_XL", t, lab_w=270,
                                       title="KL divergence from Q8_0 over 19,208 tokens, x 1000 (lower = closer)")
        if RAM64:
            charts["ram64"] = grouped("r", RAM_CATS, [("r128", "128 GB", RAM64["128"]), ("r64", "64 GB", RAM64["64"])],
                                      "Decode speed with 128 GB and with 64 GB of RAM", t, lab_w=210, vmax=135)
        for name, text in charts.items():
            (OUT / f"{name}-{mode}.svg").write_text(text, encoding="utf-8")
    print("charts written to", OUT)


if __name__ == "__main__":
    main()
