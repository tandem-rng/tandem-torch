"""Rebuild tests/cross.json from the generated cross-check headers of the submodules, which
come from tandem-cuda's core.hpp. Run after moving the pins: python tools/cross_json.py"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C_TESTS = ROOT / "external" / "tandem-c" / "tests"
CUDA_TESTS = ROOT / "external" / "tandem-cuda" / "tests"


def ints(text):
    return [int(x) for x in re.findall(r"(\d+)u(?:ll)?\b", text)]


def c_rows(text, name):
    """Rows {start, n, want[64], end_pos} of a tandem-c table."""
    body = re.search(rf"{name}\[\] = \{{(.*?)\n\}};", text, re.S).group(1)
    rows = []
    for row in re.findall(r"\{(\d+)ull, (\d+)u(?:ll)?,\s*\{(.*?)\},\s*(\d+)u\}", body, re.S):
        rows.append({"start": int(row[0]), "range": int(row[1]), "out": ints(row[2]), "end_pos": int(row[3])})
    return rows


def cuda_rows(text, name):
    body = re.search(rf"{name}\[\] = \{{(.*?)\n\}};", text, re.S).group(1)
    return [{"range": int(r), "rejected": int(rej), "out": ints(out)}
            for r, rej, out in re.findall(r"\{(\d+)u(?:ll)?, (\d+), \{(.*?)\}\}", body, re.S)]


def normal_rows(text, name):
    body = re.search(rf"{name}\[\] = \{{(.*?)\n\}};", text, re.S).group(1)
    rows = []
    for pos, n, out in re.findall(r"\{(\d+)ull, (\d+), \{(.*?)\}\}", body, re.S):
        vals = [float(x.rstrip("f")) for x in re.findall(r"[-+0-9.e]+f?", out)]
        rows.append({"pos": int(pos), "n": int(n), "out": vals[:int(n)]})
    return rows


def floats(text, name, ctype):
    body = re.search(rf"{ctype} {name}\[2 \* CROSS_NORMAL_COUNT\] = \{{(.*?)\}};", text, re.S).group(1)
    return [float(x.rstrip("f")) for x in re.findall(r"[-+0-9.e]+f?", body)]


fill = (C_TESTS / "cross_fill_below.h").read_text()
cuda = (CUDA_TESTS / "cross_fill_below.h").read_text()
cuda_normal = (CUDA_TESTS / "cross_fill_normal.h").read_text()
normal = (C_TESTS / "cross_normal.h").read_text()
key = re.search(r"CROSS_FILL_KEY\[4\] = \{(.*?)\}", cuda).group(1)
out = {
    "source": "external/tandem-c/tests and external/tandem-cuda/tests, made by tools/cross_json.py",
    # Seed 42, fills from the position `start` of each row.
    "c_fill_below32": c_rows(fill, "CROSS_FILL_U32"),
    "c_fill_below64": c_rows(fill, "CROSS_FILL_U64"),
    # The key of seed 42, K = 32, position 0.
    "cuda_key": [int(x, 16) for x in re.findall(r"0x([0-9a-f]+)u", key)],
    "cuda_below32": cuda_rows(cuda, "CROSS_BELOW32"),
    "cuda_below64": cuda_rows(cuda, "CROSS_BELOW64"),
    # Fills from the key of seed 42, K = 32, at the given position.
    "cuda_normal64": normal_rows(cuda_normal, "CROSS_NORMAL64"),
    "cuda_normal32": normal_rows(cuda_normal, "CROSS_NORMAL32"),
    # Pairs after one bool from seed 42: element 2i is the cos half, 2i + 1 the sin half.
    "normal_f64": floats(normal, "CROSS_NORMAL", "double"),
    "normal_f64_end_pos": int(re.search(r"CROSS_NORMAL_END_POS = (\d+)u", normal).group(1)),
    "normal_f32": floats(normal, "CROSS_NORMALF", "float"),
    "normal_f32_end_pos": int(re.search(r"CROSS_NORMALF_END_POS = (\d+)u", normal).group(1)),
}
(ROOT / "tests" / "cross.json").write_text(json.dumps(out, indent=1) + "\n")
