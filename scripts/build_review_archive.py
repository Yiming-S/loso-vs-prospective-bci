#!/usr/bin/env python3
"""Build the IEEE Access source-and-results supplement and checksum manifest."""
from __future__ import annotations

import hashlib
import csv
import io
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "reproducibility"
ARCHIVE = OUT / "loso_ieee_access_supplement.zip"
MANIFEST = OUT / "ieee_access_supplement_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files():
    files = [ROOT / "README.md", ROOT / "REVISION_NOTES.md", ROOT / ".gitignore", ROOT / "CITATION.cff",
             ROOT / "requirements-ieee-access.txt",
             ROOT / "requirements-results-only.txt"]
    files.extend(sorted((ROOT / "src").glob("*.py")))
    excluded_scripts = {
        "eegnet_progress.py", "run_eegnet_robustness.py",
        "write_robustness_manifest.py",
    }
    files.extend(path for path in sorted((ROOT / "scripts").glob("*.py"))
                 if path.name not in excluded_scripts)
    files.extend(sorted((ROOT / "TEST").glob("*.py")))
    paper_dependencies = {
        "main.tex", "revision_results.tex", "table_common_support.tex",
        "table_decision_impact.tex", "table_estimands.tex",
        "table_practical_impact.tex", "table_sensitivity.tex",
        "table_support.tex", "references.bib", "ieeeaccess.cls",
        "IEEEtran.cls", "IEEEtran.bst", "spotcolor.sty",
    }
    manuscript = (ROOT / "paper" / "main.tex").read_text()
    for name in re.findall(r"\\input\{([^}]+)\}", manuscript):
        paper_dependencies.add(name if Path(name).suffix else name + ".tex")
    files.extend(ROOT / "paper" / name for name in paper_dependencies)
    font_suffixes = {".pfb", ".tfm", ".fd", ".map"}
    files.extend(path for path in sorted((ROOT / "paper").iterdir())
                 if path.suffix.lower() in font_suffixes)
    for name in ("logo.png", "notaglinelogo.png", "bullet.png"):
        files.append(ROOT / "paper" / name)
    files.append(ROOT / "paper" / "main.pdf")
    files.extend(sorted((ROOT / "results").glob("*.csv")))
    files = [path for path in files if path.name != "model_ranking.csv"]
    files.extend([ROOT / "results" / "audit.json",
                  ROOT / "results" / "submitted_audit.json",
                  ROOT / "results" / "revision_summary.json",
                  ROOT / "results" / "stieger_exact_reuse_manifest.json"])
    files.extend(sorted((ROOT / "results" / "sensitivity").glob("*.csv")))
    figure_stems = {
        "fig0_protocol_logic", "fig1_gap_forest", "fig2_estimands_distribution",
        "fig3_history_by_dataset", "fig4_drift_facets", "fig5_ma_time_unit",
        "fig6_decoder_margins",
    }
    figure_stems.update(Path(name).stem for name in re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", manuscript))
    for stem in figure_stems:
        files.extend([ROOT / "results" / "figures" / f"{stem}.svg",
                      ROOT / "results" / "figures" / f"{stem}.pdf"])
    missing = sorted(str(path.relative_to(ROOT)) for path in files if not path.is_file())
    if missing:
        raise FileNotFoundError(f"required submission files are missing: {missing}")
    return sorted(set(files))


def local_path_check(files):
    blocked = re.compile(b"/" + b"Users/" + b"|/" + b"Volumes/",
                         flags=re.IGNORECASE)
    failures = []
    text_suffixes = {".py", ".tex", ".bib", ".md", ".txt", ".csv", ".json", ".cff"}
    for path in files:
        if path.suffix.lower() not in text_suffixes:
            continue
        if blocked.search(path.read_bytes()):
            failures.append(str(path.relative_to(ROOT)))
    if failures:
        raise RuntimeError(f"identifying local paths remain in: {failures}")


def archive_bytes(path):
    """Return file bytes, removing exploratory EEGNet rows from CSV outputs."""
    if path.suffix.lower() != ".csv":
        return path.read_bytes()
    text = path.read_text()
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "model" not in reader.fieldnames:
        return text.encode()
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=reader.fieldnames,
                            lineterminator="\n")
    writer.writeheader()
    for row in reader:
        if row.get("model") != "eegnet":
            writer.writerow(row)
    return stream.getvalue().encode()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads((ROOT / "results" / "submitted_audit.json").read_text())
    if audit.get("scope") != "submitted-classical" or not audit.get("ok"):
        raise RuntimeError("Run the submitted-classical audit successfully before packaging")
    files = selected_files()
    local_path_check(files)
    hashes = {}
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in files:
            name = str(path.relative_to(ROOT))
            payload = archive_bytes(path)
            archive.writestr(name, payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        archive.writestr("REPRODUCIBILITY_MANIFEST.json", json.dumps({
            "scope": "Submitted classical-model source and saved results",
            "contains_raw_eeg": False,
            "results_only_command": "python scripts/revision_analysis.py",
            "audit_command": "python scripts/audit_results.py --scope submitted-classical",
            "file_sha256": hashes,
        }, indent=2) + "\n")
    record = {
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "sha256": sha256(ARCHIVE),
        "bytes": ARCHIVE.stat().st_size,
        "files": len(files) + 1,
        "contains_raw_eeg": False,
        "scope": "IEEE Access source, pinned direct dependencies, classical-model result data, vector figures, and reproducibility code",
    }
    MANIFEST.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
