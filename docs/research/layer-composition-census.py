#!/usr/bin/env python3
"""Reproduce the reviewed #609 census from git objects, without building or running a model."""

import argparse
from collections import Counter
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import subprocess
import sys

DECLARATIONS = {
    "function_item", "function_signature_item", "struct_item", "enum_item",
    "const_item", "static_item", "type_item", "macro_definition", "trait_item", "union_item",
    "macro_invocation",
}
CLASSES = {"F", "G", "V"}
VERSIONS = {"tree-sitter": "0.25.2", "tree-sitter-rust": "0.24.2"}


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=True)
    return result.stdout


def digest(data):
    return hashlib.sha256(data).hexdigest()


def glob_pattern(pattern):
    return re.compile("^" + re.escape(pattern).replace(r"\*\*", "\x00")
                      .replace(r"\*", "[^/]*").replace("\x00", ".*") + "$")


def families_for(path, family_rows):
    candidates = [path]
    app = "crates/app/src/arch/"
    private_app = path.startswith(app) and "/" in path[len(app):]
    if private_app:
        relative = path[len(app):]
        candidates += ["crates/model/src/arch/" + relative, "crates/gpu/src/arch/" + relative]
    names = [name for name, patterns in family_rows
             if any(pattern.fullmatch(candidate) for pattern in patterns for candidate in candidates)]
    if private_app and not names:
        raise ValueError(f"Private app namespace has no existing family ownership: {path}")
    return names


def verification_file(path):
    return "/refset/" in path or "/tests/" in path or re.search(r"/gate_[^/]+\.rs$", path) is not None


def source_units(parser, path, data):
    """Assign each non-comment code-bearing physical row exactly once, first leaf wins."""
    tree = parser.parse(data)
    if tree.root_node.has_error:
        raise ValueError(f"Rust parse error: {path}")
    source_lines = data.splitlines()
    units = []

    def text(node):
        return data[node.start_byte:node.end_byte].decode("utf-8")

    def attributes(node):
        found = []
        previous = node.prev_named_sibling
        while previous is not None and previous.type in ("attribute_item", "line_comment", "block_comment"):
            if previous.type == "attribute_item":
                found.append(previous)
            previous = previous.prev_named_sibling
        return found

    def impl_name(node):
        ty, trait = node.child_by_field_name("type"), node.child_by_field_name("trait")
        label = text(ty) if ty else "impl"
        if trait:
            label = text(trait) + " for " + label
        return re.sub(r"\s+", " ", label)

    def visit(node, context=""):
        attrs = attributes(node)
        attr_text = re.sub(r"\s+", "", " ".join(text(a) for a in attrs))
        test = re.search(r"#\[test\]|#\[cfg\(test\)\]", attr_text) is not None
        name_node = node.child_by_field_name("name")
        name = text(name_node) if name_node else node.type
        if node.type == "macro_invocation":
            name = "@" + text(node.child_by_field_name("macro"))
        test_container = (node.type == "mod_item" and (test or name in ("test", "tests"))) or (
            node.type == "impl_item" and test)
        if node.type in DECLARATIONS or test_container:
            start = min([node.start_byte] + [a.start_byte for a in attrs])
            symbol = context + ("impl " + impl_name(node) if node.type == "impl_item" else name)
            units.append({"start_byte": start, "end_byte": node.end_byte,
                          "start": data[:start].count(b"\n") + 1,
                          "end": node.end_point.row + (node.end_point.column != 0),
                          "symbol": symbol, "auto": "V" if test or test_container else None,
                          "rows": set()})
            return
        if node.type == "impl_item":
            context += impl_name(node) + "::"
        elif node.type == "mod_item":
            context += name + "::"
        for child in node.named_children:
            visit(child, context)

    if verification_file(path):
        units.append({"start_byte": 0, "end_byte": len(data), "start": 1, "end": len(source_lines),
                      "symbol": "@file", "auto": "V", "rows": set()})
    else:
        visit(tree.root_node)
    units.sort(key=lambda u: u["start_byte"])
    scaffold = {"start_byte": 0, "end_byte": len(data), "start": 1, "end": len(source_lines),
                "symbol": "@scaffolding", "auto": "G", "rows": set()}
    code_rows = {}

    def leaves(node):
        if node.type in ("line_comment", "block_comment"):
            return
        if not node.children:
            for row in range(node.start_point.row, node.end_point.row + (node.end_point.column != 0)):
                if row < len(source_lines) and source_lines[row].strip():
                    code_rows.setdefault(row, node.start_byte)
            return
        for child in node.children:
            leaves(child)

    leaves(tree.root_node)
    index = 0
    for row, offset in sorted(code_rows.items()):
        while index < len(units) and units[index]["end_byte"] <= offset:
            index += 1
        owner = units[index] if index < len(units) and units[index]["start_byte"] <= offset < units[index]["end_byte"] else scaffold
        owner["rows"].add(row + 1)
    records = []
    for unit in units + [scaffold]:
        rows = sorted(unit.pop("rows"))
        if rows:
            records.append({"path": path, "symbol": unit["symbol"], "start": unit["start"],
                            "end": unit["end"], "lines": len(rows), "code_rows": rows, "auto": unit["auto"]})
    if sum(u["lines"] for u in records) != len(code_rows):
        raise ValueError(f"Row partition does not cover {path}")
    return records


def analyze(root, data):
    for package, wanted in VERSIONS.items():
        actual = metadata.version(package)
        if actual != wanted:
            raise ValueError(f"{package}: use {wanted}, found {actual}")
    import tree_sitter
    import tree_sitter_rust

    base = data["base"]
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("The census base must be a full commit ID")
    resolved = git(root, "rev-parse", base + "^{commit}").decode().strip()
    if resolved != base:
        raise ValueError("The census base does not resolve to its recorded commit")
    parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_rust.language()))
    cache = {}

    def blob(path, expected):
        content = git(root, "show", f"{base}:{path}")
        if digest(content) != expected:
            raise ValueError(f"Source SHA-256 mismatch: {path}")
        return content

    family_text = blob("tools/families.tsv", data["families_sha256"]).decode()
    family_rows = []
    for line in family_text.splitlines():
        if line and not line.startswith("#"):
            fields = line.split("\t")
            family_rows.append((fields[0], [glob_pattern(p) for p in fields[2].split()]))
    family_names = [name for name, _ in family_rows]
    registry = blob("tools/arch-registry.tsv", data["registry_sha256"]).decode()
    canonical = [line.split("\t")[0].split(",")[0] for line in registry.splitlines()
                 if line and not line.startswith("#") and not line.startswith("names\t")]
    if len(set(family_names)) != len(family_names) or set(canonical) - set(family_names):
        raise ValueError("Family table duplicates or misses registered canonical architectures")
    tracked = git(root, "ls-tree", "-r", "-z", "--name-only", base).decode().split("\0")
    ownership = {path: families_for(path, family_rows) for path in tracked if path.endswith(".rs")}
    ownership = {path: families for path, families in ownership.items() if families}
    if set(ownership) != set(data["files"]):
        raise ValueError("The reviewed file set differs from the base's family-owned Rust file set")
    totals = Counter({kind: 0 for kind in sorted(CLASSES)})
    families = {name: Counter({kind: 0 for kind in sorted(CLASSES)}) for name in family_names}
    all_units = []
    for path, names in sorted(ownership.items()):
        policy = data["files"][path]
        if policy["default"] not in CLASSES or not policy["reason"]:
            raise ValueError(f"Missing semantic classification/reason: {path}")
        records = source_units(parser, path, blob(path, policy["sha256"]))
        cache[path] = records
        reviewed = {u["symbol"] for u in records if u["auto"] is None}
        overrides = policy.get("overrides", {})
        if set(overrides) - reviewed or set(overrides.values()) - CLASSES:
            raise ValueError(f"Unknown classification override: {path}: {sorted(set(overrides) - reviewed)}")
        for unit in records:
            if unit["auto"] == "V" or policy["default"] == "V":
                kind = "V"
            elif unit["auto"] == "G":
                kind = "G"
            else:
                kind = overrides.get(unit["symbol"], policy["default"])
            unit["class"] = kind
            totals[kind] += unit["lines"]
            for name in names:
                families[name][kind] += unit["lines"]
            all_units.append(unit)
    counts = {"families": {name: {**dict(count), "total": sum(count.values()),
                                  "files": sum(name in names for names in ownership.values())}
                           for name, count in families.items()},
              "deduplicated": {**dict(totals), "total": sum(totals.values()),
                               "files": len(ownership), "units": len(all_units)}}
    if counts != data["counts"]:
        raise ValueError("Computed counts differ from the reviewed census data")

    def resolve(selector):
        path, symbol = selector.split(":", 1)
        if path not in cache:
            expected = data["supporting_files"].get(path)
            if expected is None:
                raise ValueError(f"Unrecorded supporting source: {path}")
            cache[path] = source_units(parser, path, blob(path, expected))
        found = [u for u in cache[path] if u["symbol"] == symbol]
        if len(found) != 1:
            raise ValueError(f"Twin/lift item must resolve exactly once: {selector} ({len(found)} matches)")
        unit = found[0]
        return {key: unit[key] for key in ("path", "symbol", "start", "end", "lines")}

    twins = [{**twin, "left_item": resolve(twin["left"]), "right_item": resolve(twin["right"])}
             for twin in data["twins"]]
    used, lifts = set(), []
    for lift in data["lifts"]:
        selected = lift["sources"]
        if len(set(selected)) != len(selected) or used.intersection(selected):
            raise ValueError(f"Lift sources overlap: {lift['id']}")
        used.update(selected)
        items = [resolve(selector) for selector in selected]
        before = sum(item["lines"] for item in items)
        common, bindings = lift["common_budget"], lift["bindings_budget"]
        for interval in (common, bindings):
            if len(interval) != 2 or not 0 <= interval[0] <= interval[1]:
                raise ValueError(f"Invalid forecast budget: {lift['id']}")
        removed = [before - common[1] - bindings[1], before - common[0] - bindings[0]]
        if removed != lift["predicted_removed"]:
            raise ValueError(f"Removal arithmetic differs: {lift['id']}")
        lifts.append({**lift, "before": before, "items": items})
    return {"base": base, "parser_versions": VERSIONS, "canonical_architectures": canonical,
            "counts": counts, "twins": twins, "lifts": lifts}


def markdown(report):
    print("| Family | F | G | V | Code rows | Files |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, count in report["counts"]["families"].items():
        print(f"| {name} | {count['F']} | {count['G']} | {count['V']} | {count['total']} | {count['files']} |")
    count = report["counts"]["deduplicated"]
    print(f"| **Deduplicated union** | **{count['F']}** | **{count['G']}** | **{count['V']}** | **{count['total']}** | **{count['files']}** |")
    print("\n| Lift | Existing code rows | New common owner [forecast] | Bindings [forecast] | Net removed [derived forecast] |")
    print("|---|---:|---:|---:|---:|")
    for lift in report["lifts"]:
        interval = lambda values: f"{values[0]}–{values[1]}"
        print(f"| {lift['id']}: {lift['title']} | {lift['before']} | {interval(lift['common_budget'])} | {interval(lift['bindings_budget'])} | {interval(lift['predicted_removed'])} |")
    print(f"\nVerified {count['units']} item/scaffolding units; {len(report['twins'])} source-resolved twin pairs; all canonical architecture rows covered.")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2], help="git repository containing the frozen base")
    cli.add_argument("--data", type=Path, default=Path(__file__).with_suffix(".json"))
    cli.add_argument("--json", action="store_true", help="emit the verified report, including exact twin/lift spans")
    args = cli.parse_args()
    try:
        report = analyze(args.root, json.loads(args.data.read_text()))
    except (OSError, ValueError, metadata.PackageNotFoundError, subprocess.CalledProcessError) as error:
        print(f"census refused: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        markdown(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
