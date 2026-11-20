"""Builds the self-contained beginner guide: every diagram is embedded inside the Markdown file.

Edit build/guide_source.md (normal image links like images/01_x.png), regenerate diagrams with
make_diagrams.py / make_diagrams_phase2.py if needed, then run:  python guide/build/build_guide.py
Output: guide/Credit_Card_Fraud_Beginner_Guide.md (one file, no external image files needed).
"""
import base64
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "guide_source.md")
OUT = os.path.join(os.path.dirname(HERE), "Credit_Card_Fraud_Beginner_Guide.md")

text = open(SRC, encoding="utf-8").read()
missing = []


def embed(m):
    alt, path = m.group(1), m.group(2)
    full = os.path.join(HERE, path)
    if not os.path.exists(full):
        missing.append(path)
        return m.group(0)
    data = base64.b64encode(open(full, "rb").read()).decode("ascii")
    return f"![{alt}](data:image/png;base64,{data})"


out = re.sub(r"!\[([^\]]*)\]\((images/[^)]+\.png)\)", embed, text)
if missing:
    sys.exit(f"missing images: {missing}")
header = ("<!-- Self-contained guide: all diagrams are embedded in this file. "
          "Edit guide/build/guide_source.md and run guide/build/build_guide.py to rebuild. -->\n\n")
open(OUT, "w", encoding="utf-8").write(header + out)
print(f"embedded {len(re.findall(r'data:image/png;base64', out))} images -> {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
