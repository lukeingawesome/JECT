"""Section-structured findings text to utterances.

Input format, one finding per bullet under a bracketed section header:

    [Lungs]
    - Millimetric nodules in both lungs.
    [Pleura]
    - No pleural effusion.

Each bullet becomes one utterance with its section, an anatomy name the trainer
maps to an anatomy id, and two regex flags (negated, uncertain).
"""
from __future__ import annotations

import re

# Slot order of the section targets; must equal ject.data.CANONICAL_SECTIONS.
SECTIONS = ["Lungs", "Airways & Trachea", "Pleura", "Mediastinum & Hila", "Cardiovascular",
            "Chest Wall", "Bones / Spine", "Upper abdomen", "Lower neck", "Breast & Axilla",
            "Others"]

# Header spellings that map onto a canonical section.
SECTION_ALIASES = {"Trachea & Airways": "Airways & Trachea", "Upper Abdomen": "Upper abdomen",
                   "Lower Neck": "Lower neck", "Breast": "Breast & Axilla"}

# Values are keys of ject.data.ANATOMY_NAME_TO_ID.
ANATOMY = {"Lungs": "lung", "Airways & Trachea": "airways", "Pleura": "pleura",
           "Mediastinum & Hila": "mediastinum", "Cardiovascular": "cardiovascular",
           "Chest Wall": "chest_wall", "Bones / Spine": "bones", "Upper abdomen": "upper_abdomen",
           "Lower neck": "lower_neck", "Breast & Axilla": "breast_axilla", "Others": "other"}
ANATOMY.update({alias: ANATOMY[canon] for alias, canon in SECTION_ALIASES.items()})

KNOWN_HEADERS = set(SECTIONS) | set(SECTION_ALIASES)

HEADER = re.compile(r"^\[([^\]]+)\]\s*$")
NEG = re.compile(r"^\s*(no\b|without\b|not\s+(seen|identified|visualized|detected|evident)|absence of|unremarkable|"
                 r"normal\b|negative for|neither\b|resolved|disappeared|cleared)|"
                 r"\bno\s+(definite|significant|evidence|abnormal|enlarged|focal|other|new)\b", re.I)
UNC = re.compile(r"\b(suspicious|suspect(ed)?|possibl[ey]|may\b|might\b|cannot (be )?exclude[d]?|r/o\b|rule out|ddx|"
                 r"differential|indeterminate|equivocal|questionable|versus|vs\.|less likely|more likely|likely)\b", re.I)


def header_of(line: str) -> str | None:
    """Section name if `line` is a header, else None.

    A line with a missing bracket ("Pleura]" or bare "Pleura") counts as a header
    when the stripped text is a known section name.
    """
    m = HEADER.match(line)
    if m:
        return m.group(1).strip()
    bare = line.strip("[] \t")
    return bare if bare in KNOWN_HEADERS else None


def has_header(text: str) -> bool:
    return any(header_of(line.strip()) for line in str(text).splitlines())


def canonical_section(name: str | None) -> str:
    """Map a section name onto SECTIONS; unknown names go to Others."""
    s = (name or "Others").strip()
    s = SECTION_ALIASES.get(s, s)
    return s if s in SECTIONS else "Others"


def parse(text: str) -> list[dict]:
    """Findings text to [{text, section, anatomy, negated, uncertain}], one per bullet.

    Bullet text is kept verbatim. A non-bullet line under a header continues the
    previous bullet. Bullets before any header go to Others.
    """
    out, sec = [], None
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        h = header_of(line)
        if h is not None:
            sec = h
            continue
        if line.startswith("- "):
            # drop a stray "[" left by a header split across lines
            body = re.sub(r"\s*\[\s*$", "", line[2:]).strip()
            if body:
                s = sec or "Others"
                out.append({"text": body, "section": s, "anatomy": ANATOMY.get(s, "other"),
                            "negated": bool(NEG.search(body)), "uncertain": bool(UNC.search(body))})
        elif out and sec is not None:
            out[-1]["text"] = (out[-1]["text"] + " " + line).strip()
    return out
