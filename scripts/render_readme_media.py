"""Render README visuals from a saved, public BANKING77 analysis.

Requires NumPy and Pillow. Does not query a database, run a model, or fabricate
analysis results. GIF timing controls the presentation, not computation speed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont


W, H = 1280, 720
BG = "#06121c"
PANEL = "#0b1d2a"
LINE = "#1c3948"
WHITE = "#eef4f8"
MUTED = "#97afbf"
CYAN = "#20d8e9"
COLORS = [CYAN, "#38d6ae", "#f7c768", "#b29aff", "#fa829e", "#68b5fb",
          "#a3dc74", "#f7a96d", "#df9ce8", "#83d9d1", "#d7d676", "#8baff0",
          "#eaa2a2", "#b8c4d6", "#60cab0", "#d2a7ef", "#9abf80", "#75d7ee"]


def font(size, bold=False):
    candidates = [
        f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
    ]
    for name in candidates:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def box(draw, xy, fill=PANEL, outline=LINE, radius=14):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=1)


def text(draw, xy, value, size=22, color=WHITE, bold=False):
    draw.text(xy, str(value), font=font(size, bold), fill=color)


def header(draw, title, subtitle):
    text(draw, (36, 24), "E I G E N L E N S   /   PUBLIC DATA DEMO", 15, CYAN, True)
    text(draw, (36, 62), title, 34, WHITE, True)
    text(draw, (36, 108), subtitle, 19, MUTED)


def scene(data, angle, focus=None):
    """Project the actual saved PCA coordinates; retain every selected point."""
    values = np.asarray([p["pca_3d"] for p in data["points"]], dtype=float)
    # Equal per-axis normalization is only for displaying the 3D scene.
    values /= max(float(np.max(np.abs(values))), 1e-9)
    azimuth = np.radians(angle)
    elevation = np.radians(22)
    rx = np.array([[np.cos(azimuth), -np.sin(azimuth), 0],
                   [np.sin(azimuth), np.cos(azimuth), 0], [0, 0, 1]])
    ry = np.array([[1, 0, 0], [0, np.cos(elevation), -np.sin(elevation)],
                   [0, np.sin(elevation), np.cos(elevation)]])
    rotated = values @ rx.T @ ry.T
    projected = np.column_stack((rotated[:, 0] * 375 + 465,
                                 -rotated[:, 1] * 220 + 409))
    ids = [p["cluster_id"] for p in data["points"]]
    palette = {g["cluster_id"]: COLORS[i % len(COLORS)]
               for i, g in enumerate(g for g in data["groups"] if g["cluster_id"] != -1)}
    im = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(im)
    for x in range(70, 910, 70):
        draw.line((x, 174, x, 635), fill="#102735")
    for y in range(187, 640, 56):
        draw.line((42, y, 914, y), fill="#102735")
    order = sorted(range(len(ids)), key=lambda i: (ids[i] != -1, rotated[i, 2]))
    for i in order:
        cid = ids[i]
        x, y = projected[i]
        if focus is not None and cid != focus:
            color, radius = "#203746", 1.6
        elif cid == -1:
            color, radius = "#3b5364", 1.8
        else:
            color, radius = palette[cid], 3.6 if focus is None else 4.8
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    text(draw, (42, 650), "3D PCA view  /  every selected record shown", 17, MUTED)
    text(draw, (42, 680), "Saved analysis visualization. Animation is not a runtime benchmark.", 15, MUTED)
    return im, draw, palette


def overview(data, angle):
    im, draw, palette = scene(data, angle)
    header(draw, "From stored vectors to discovered structure",
           'BANKING77  /  saved selection for "payment"  /  DBSCAN + BERTopic')
    analysis = data["analysis"]
    for top, value, label in [(176, len(data["points"]), "SELECTED RECORDS"),
                              (286, analysis["cluster_count"], "CLUSTERS"),
                              (396, analysis["noise_count"], "UNASSIGNED")]:
        box(draw, (950, top, 1242, top+96))
        text(draw, (970, top+12), f"{value:,}", 34, WHITE, True)
        text(draw, (970, top+59), label, 14, MUTED, True)
    text(draw, (953, 522), "LARGEST TOPIC", 14, CYAN, True)
    first = next(g for g in data["groups"] if g["cluster_id"] != -1)
    for i, line in enumerate(textwrap.wrap(first["label"], 21)):
        text(draw, (953, 550+i*26), line, 22, WHITE, True)
    text(draw, (953, 610), "Gray = DBSCAN noise", 16, MUTED)
    return im


def topic_view(data, group):
    im, draw, palette = scene(data, 28, focus=group["cluster_id"])
    header(draw, "Investigate a discovered topic",
           "Same saved analysis  /  topic terms and representative source text")
    # Give source evidence room without making the chart a thumbnail.
    box(draw, (920, 174, 1242, 635))
    color = palette[group["cluster_id"]]
    text(draw, (942, 195), f'CLUSTER {group["cluster_id"]}  /  {group["count"]} RECORDS', 14, color, True)
    y = 230
    for line in textwrap.wrap(group["label"], 19):
        text(draw, (942, y), line, 25, WHITE, True)
        y += 30
    text(draw, (942, y+12), "BERTOPIC PHRASES", 13, MUTED, True)
    y += 40
    for term in group["terms"][:4]:
        text(draw, (942, y), term["term"], 19, color)
        y += 27
    text(draw, (942, y+20), "REPRESENTATIVE EXAMPLE", 13, MUTED, True)
    y += 52
    example = min(group["examples"], key=lambda e: len(e["text"]))
    lines = textwrap.wrap(example["text"], 25)
    for line in lines[:5]:
        text(draw, (942, y), line, 18, WHITE)
        y += 24
    if len(lines) > 5:
        text(draw, (942, y), "...", 18, MUTED)
    return im


def gif(frames, path, durations):
    # Use every scene to retain the colors of topics absent from the first frame.
    sheet = Image.new("RGB", (W, H * len(frames)))
    for i, frame in enumerate(frames):
        sheet.paste(frame, (0, H * i))
    palette = sheet.quantize(colors=224)
    indexed = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames]
    indexed[0].save(path, save_all=True, append_images=indexed[1:],
                    duration=durations, loop=0, disposal=2, optimize=True)


def hero(data, path):
    colors = {g["cluster_id"]: COLORS[i % len(COLORS)]
              for i, g in enumerate(g for g in data["groups"] if g["cluster_id"] != -1)}
    points = np.asarray([p["pca_3d"][:2] for p in data["points"]])
    minimum, span = points.min(axis=0), np.ptp(points, axis=0)
    circles = []
    for p, xy in zip(data["points"], points):
        x, y = (xy-minimum)/span
        cid = p["cluster_id"]
        circles.append(f'<circle cx="{890+x*420:.1f}" cy="{115+y*200:.1f}" r="{1.3 if cid == -1 else 2.8}" fill="{colors.get(cid, "#355364")}"/>')
    path.write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="440" viewBox="0 0 1440 440" role="img" aria-labelledby="title desc">
<title id="title">EigenLens - Agents can retrieve. Now let them discover.</title>
<desc id="desc">A discovery layer over existing vector data. Keep your agent, keep RAG, add discovery. The graphic uses a saved public BANKING77 selection.</desc>
<rect width="1440" height="440" rx="22" fill="#06121c"/>
<rect x="1" y="1" width="1438" height="438" rx="21" fill="none" stroke="#1c3948"/>
<g font-family="Inter,Arial,sans-serif">
<text x="52" y="48" fill="#20d8e9" font-size="18" font-weight="700" letter-spacing="4">EIGENLENS</text>
<text x="52" y="133" fill="#eef4f8" font-size="58" font-weight="700">Agents can retrieve.</text>
<text x="52" y="202" fill="#20d8e9" font-size="58" font-weight="700">Now let them discover.</text>
<text x="54" y="257" fill="#b2c5d2" font-size="23">A discovery layer for the vector data you already have.</text>
<g font-size="13" font-weight="700" letter-spacing="1.2">
<rect x="54" y="299" width="172" height="38" rx="8" fill="#0c2b29" stroke="#38d6ae"/>
<text x="71" y="324" fill="#8fe6ce">KEEP YOUR AGENT</text>
<rect x="240" y="299" width="112" height="38" rx="8" fill="#302819" stroke="#b49145"/>
<text x="258" y="324" fill="#f7c768">KEEP RAG</text>
<rect x="366" y="299" width="166" height="38" rx="8" fill="#0b2634" stroke="#20d8e9"/>
<text x="384" y="324" fill="#20d8e9">ADD DISCOVERY</text></g>
<text x="54" y="391" fill="#97afbf" font-size="17">Existing embeddings. Clusters. Topics. Evidence for your agent.</text>
<rect x="838" y="58" width="553" height="319" rx="16" fill="#0b1d2a" stroke="#1c3948"/>
<text x="865" y="91" fill="#20d8e9" font-size="13" letter-spacing="2">FROM NEIGHBORS TO STRUCTURE</text>
''' + ''.join(circles) + '''
<text x="866" y="354" fill="#97afbf" font-size="14">BANKING77 / saved payment selection</text>
</g></svg>''')


def workflow(path):
    path.write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="430" viewBox="0 0 1440 430" role="img" aria-labelledby="title desc">
<title id="title">Add discovery beside retrieval and existing action tools</title>
<desc id="desc">An existing agent keeps its retrieval and action tools, and calls EigenLens through MCP or Python. Retrieval and discovery use existing vector data.</desc>
<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9" fill="none" stroke="#6a899b"/></marker></defs>
<rect width="1440" height="430" rx="18" fill="#06121c"/>
<g fill="none" stroke="#496a7b" stroke-width="2" marker-end="url(#arrow)">
<path d="M720 105 V136 H250 V174"/><path d="M720 105 V174"/><path d="M720 105 V136 H1190 V174"/>
<path d="M720 340 V294"/><path d="M250 340 V294"/></g>
<g font-family="Inter,Arial,sans-serif" text-anchor="middle">
<rect x="485" y="22" width="470" height="83" rx="12" fill="#0b2634" stroke="#20d8e9"/>
<text x="720" y="58" fill="#eef4f8" font-size="25" font-weight="700">Your existing agent</text>
<text x="720" y="83" fill="#97afbf" font-size="17">Same model, prompts and orchestration</text>
<rect x="60" y="175" width="380" height="119" rx="12" fill="#191e23" stroke="#b49145"/>
<text x="250" y="211" fill="#f7c768" font-size="14" letter-spacing="2">RETRIEVE</text>
<text x="250" y="244" fill="#eef4f8" font-size="25" font-weight="700">Your existing RAG</text>
<text x="250" y="272" fill="#97afbf" font-size="17">Find evidence for a known question</text>
<rect x="530" y="175" width="380" height="119" rx="12" fill="#0b2634" stroke="#20d8e9" stroke-width="2"/>
<text x="720" y="211" fill="#20d8e9" font-size="14" letter-spacing="2">DISCOVER / MCP OR PYTHON SDK</text>
<text x="720" y="244" fill="#eef4f8" font-size="25" font-weight="700">EigenLens</text>
<text x="720" y="272" fill="#97afbf" font-size="17">Clusters, topics and representative records</text>
<rect x="1000" y="175" width="380" height="119" rx="12" fill="#171e30" stroke="#a28ada"/>
<text x="1190" y="211" fill="#b29aff" font-size="14" letter-spacing="2">ACT</text>
<text x="1190" y="244" fill="#eef4f8" font-size="25" font-weight="700">Your existing tools</text>
<text x="1190" y="272" fill="#97afbf" font-size="17">SQL, APIs and application workflows</text>
<rect x="60" y="340" width="850" height="62" rx="10" fill="#0b1d2a" stroke="#1c3948"/>
<text x="485" y="378" fill="#b2c5d2" font-size="20">Existing vector data + text + metadata</text>
</g></svg>''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/assets/demo/banking77-payments.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    counts = Counter(p["cluster_id"] for p in data["points"])
    assert len(data["points"]) == data["source"]["selected_records"]
    assert counts[-1] == data["analysis"]["noise_count"]
    assert len(counts)-1 == data["analysis"]["cluster_count"]
    assert {g["cluster_id"]: g["count"] for g in data["groups"]} == dict(counts)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = [overview(data, angle) for angle in np.linspace(0, 360, args.frames, endpoint=False)]
    frames[0].save(args.output/"banking77-clusters.png", optimize=True)
    gif(frames, args.output/"banking77-clusters.gif", 160)
    focus = [g for g in data["groups"] if g["cluster_id"] != -1][:3]
    gif([topic_view(data, g) for g in focus], args.output/"banking77-topics.gif", 2800)
    hero(data, args.output/"eigenlens-hero.svg")
    workflow(args.output/"discovery-workflow.svg")
    for name in ["banking77-clusters.gif", "banking77-topics.gif", "banking77-clusters.png"]:
        print(f"{name}: {(args.output/name).stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
