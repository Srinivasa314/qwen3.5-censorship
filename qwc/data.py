"""Prompts dataset and the overgeneralization exclusion list.

200 prompts in four classes of 50 each:
    prc_sensitive       — Tiananmen + ccp_other (Xi/Taiwan/Xinjiang/HK/Tibet/Falun)
    neutral_political   — structurally-matched non-PRC political controls
    harmful             — AdvBench-style cybercrime/fraud/weapons/drugs/self-harm
    harmless            — math, code, DIY, science

Six IDs flagged in `_meta.overgeneralization_and_anomalies` are excluded from
direction-extraction class means and from probe training (their baseline
behaviour is mis-tagged because the writer directions overgeneralize).
"""
import json
from .config import PROMPTS_PATH

SETS = ("prc_sensitive", "neutral_political", "harmful", "harmless")


def load() -> dict:
    """Return the full prompts.json blob (with _meta)."""
    with open(PROMPTS_PATH) as f:
        return json.load(f)


def by_class() -> dict[str, list[dict]]:
    """Return {class_name: [item, ...]} for the four data classes."""
    blob = load()
    return {s: blob[s] for s in SETS}


def all_items() -> list[dict]:
    """Flatten into a list with a `class` field added per item."""
    out = []
    for cls, items in by_class().items():
        for it in items:
            out.append({**it, "class": cls})
    return out


def excluded_ids() -> set[str]:
    """The six IDs to drop from class means and probe training."""
    meta = load()["_meta"]["overgeneralization_and_anomalies"]
    out = set()
    for group in ("dprc_overgeneralization", "drefuse_overgeneralization", "baseline_anomaly"):
        for entry in meta.get(group, []):
            out.add(entry["id"])
    return out


def by_id() -> dict[str, dict]:
    """Map id -> item (with `class` field)."""
    return {it["id"]: it for it in all_items()}


def class_means_groups() -> dict[str, list[str]]:
    """ID groups for diff-of-means contrasts. Filters excluded IDs.

    Returns:
        tiananmen — prc_sensitive prompts with topic=tiananmen
        prc_other — prc_sensitive prompts with topic != tiananmen
        all_prc   — union of the two
        neutral   — neutral_political
        harmful   — harmful
        harmless  — harmless
    """
    excl = excluded_ids()
    classes = by_class()
    groups: dict[str, list[str]] = {
        "tiananmen": [], "prc_other": [], "all_prc": [],
        "neutral":   [], "harmful":   [], "harmless":  [],
    }
    for it in classes["prc_sensitive"]:
        if it["id"] in excl:
            continue
        groups["all_prc"].append(it["id"])
        if it["topic"] == "tiananmen":
            groups["tiananmen"].append(it["id"])
        else:
            groups["prc_other"].append(it["id"])
    for it in classes["neutral_political"]:
        if it["id"] not in excl:
            groups["neutral"].append(it["id"])
    for it in classes["harmful"]:
        if it["id"] not in excl:
            groups["harmful"].append(it["id"])
    for it in classes["harmless"]:
        if it["id"] not in excl:
            groups["harmless"].append(it["id"])
    return groups
