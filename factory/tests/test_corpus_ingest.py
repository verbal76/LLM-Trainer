import json

import pytest
from corpus_helpers import make_docx, make_pdf

from llmtrainer import corpus
from llmtrainer import pipeline as pl
from llmtrainer.extract import extract
from llmtrainer.extract.clean import clean
from llmtrainer.hashing import sha256_bytes
from llmtrainer.ingest import ingest_file
from llmtrainer.schemas import RightsInfo
from llmtrainer.terms import extract_terms

RIGHTS = RightsInfo(status="owned", permitted_training="yes", permitted_commercial="yes", permitted_redistribution="yes", evidence="test")
PROSE = "The coolant pump must be inspected every hundred hours of operation to prevent seal failure. "


def test_markdown_heading_hierarchy_and_exact_offsets():
    md = "# Pump\n\nIntro text about pumps here.\n\n## Seals\n\nSeal text is here and is long enough.\n"
    ex = extract("a.md", md.encode())
    paras = [b for b in ex.blocks if b.kind == "paragraph"]
    assert paras[1].section_path == ("Pump", "Seals")
    assert md[paras[1].start : paras[1].end] == "Seal text is here and is long enough."
    assert md.encode()[paras[1].byte_start : paras[1].byte_end] == b"Seal text is here and is long enough."


def test_markdown_table_is_structured_and_reference():
    md = "# T\n\n| Part | Torque |\n|---|---|\n| Bolt A | 25 Nm |\n| Bolt B | 40 Nm |\n"
    out = ingest_file(md.encode(), filename="t.md", origin="o", rights=RIGHTS, ingested_on="2026-10-05", min_chars=5)
    tab = [c for c in out.chunks if c.meta["kind"] == "table"]
    assert tab and tab[0].meta["role"] == "reference"
    assert tab[0].meta["table"]["header"] == ["Part", "Torque"] and tab[0].meta["table"]["rows"][1] == ["Bolt B", "40 Nm"]


def test_docx_headings_paragraphs_tables():
    raw = make_docx([("h1", "Engine"), ("p", PROSE), ("h2", "Valves"), ("p", "Valve clearance is checked cold."),
                     ("table", [["Valve", "Gap"], ["Intake", "0.10 mm"]])])
    ex = extract("m.docx", raw)
    assert ex.extractor == "docx_stdlib" and not ex.issues
    kinds = [b.kind for b in ex.blocks]
    assert kinds == ["heading", "paragraph", "heading", "paragraph", "table"]
    assert ex.blocks[3].section_path == ("Engine", "Valves")
    assert ex.blocks[4].rows == [["Intake", "0.10 mm"]]
    b = ex.blocks[1]
    assert ex.stream[b.start : b.end] == PROSE.strip()


def test_corrupt_encrypted_unsupported_empty_do_not_crash():
    assert extract("x.docx", b"not a zip").issues[0].code == "corrupt_file"
    assert extract("x.docx", b"\xd0\xcf\x11\xe0" + b"0" * 20).issues[0].code == "encrypted_or_legacy"
    assert extract("x.exe", b"MZ").issues[0].code == "unsupported_format"
    assert extract("x.txt", b"  \n").issues[0].code == "empty_file"
    assert extract("x.txt", b"\xff\xfe\x00bad").issues[0].code == "undecodable_text"
    assert extract("x.json", b"{broken").issues[0].code == "corrupt_structured"
    assert extract("x.pdf", b"%PDF-1.4 garbage").issues[0].code in ("corrupt_file", "missing_dependency")


def test_csv_tsv_json_jsonl_records_are_reference_tables():
    csv_ = b"part,torque\nbolt a,25\nbolt b,40\n"
    ex = extract("t.csv", csv_)
    assert ex.blocks[0].header == ["part", "torque"] and len(ex.blocks[0].rows) == 2
    assert extract("t.tsv", b"a\tb\n1\t2\n").blocks[0].rows == [["1", "2"]]
    js = extract("t.json", json.dumps({"items": [{"p": "x", "v": 1}, {"p": "y", "v": 2}]}).encode())
    assert js.blocks[0].header == ["p", "v"] and js.blocks[0].rows[1] == ["y", "2"]
    jl = extract("t.jsonl", b'{"p":"x"}\n{"p":"y"}\n')
    assert jl.blocks[0].rows == [["x"], ["y"]]
    assert jl.stream[jl.blocks[0].start : jl.blocks[0].end].startswith('{"p":"x"}')
    out = ingest_file(csv_, filename="t.csv", origin="o", rights=RIGHTS, ingested_on="2026-10-05", min_chars=5)
    assert {c.meta["role"] for c in out.chunks} == {"reference"}


def _pump_pages():
    # running header + page-number footer + hyphenated word across a line break
    pages = []
    for i in range(1, 5):
        name = ["Cooling", "Lubrication", "Ignition", "Exhaust"][i - 1]
        pages.append(["ACME Service Manual", "1.%d %s" % (i, name), "The %s unit must be inspected every hundred" % name.lower(), "hours of opera-", "tion to prevent %s failure." % name.lower(), "Page %d" % i])
    return pages


def test_cleaning_noise_hyphenation_whitespace_recorded():
    from llmtrainer.extract.models import Block, Extraction, Line

    ex = Extraction("fake", "1", "text/plain", "", False, line_based=True, detect_headings=True, pages_total=4)
    pos = 0
    for pno, lines in enumerate(_pump_pages(), 1):
        ls = []
        for t in lines:
            ls.append(Line(t, pos, pos + len(t)))
            pos += len(t) + 1
        ex.blocks.append(Block("paragraph", "\n".join(lines), ls[0].start, ls[-1].end, page=pno, lines=ls))
    blocks, rep = clean(ex)
    text = " ".join(b.text for b in blocks)
    assert "ACME" not in text and "Page 1" not in text
    assert "operation to prevent" in text
    names = {t["name"] for t in rep.transformations}
    assert {"remove_repeated_header_footer", "repair_hyphenation", "heuristic_heading_detection"} <= names
    assert any(b.kind == "heading" and b.section_path == (b.text,) for b in blocks)


def test_ingest_provenance_and_transformations():
    txt = ("# Pumps\n\n" + PROSE * 3 + "\n\n## Seals\n\n" + "Seals wear out with heat and age over time. " * 3 + "\n").encode()
    out = ingest_file(txt, filename="pumps.md", origin="o", rights=RIGHTS, ingested_on="2026-10-05")
    rec = out.record
    assert rec.sha256 == sha256_bytes(txt) and rec.source_id.startswith("src-")
    assert any(t.name.startswith("extract:") for t in rec.transformations)
    assert {c.section for c in rec.chunks} == {"Pumps", "Pumps > Seals"}
    again = ingest_file(txt, filename="pumps.md", origin="o", rights=RIGHTS, ingested_on="2026-10-05")
    assert [c.record.chunk_id for c in again.chunks] == [c.record.chunk_id for c in out.chunks]
    assert [c.record.text_sha256 for c in again.chunks] == [c.record.text_sha256 for c in out.chunks]


def test_batch_ingest_dedup_issues_and_raw_copy(tmp_path):
    pl.init_project(tmp_path / "p", "Corp", "d")
    ws = pl.Workspace(tmp_path / "p")
    src = tmp_path / "in"
    src.mkdir()
    (src / "a.txt").write_bytes((PROSE * 4).encode())
    (src / "a_copy.txt").write_bytes((PROSE * 4).encode())
    (src / "m.docx").write_bytes(make_docx([("h1", "Engine"), ("p", PROSE * 3)]))
    (src / "t.csv").write_bytes(b"part,torque\nbolt a,25\n")
    (src / "bad.docx").write_bytes(b"junk")
    (src / "x.bin").write_bytes(b"\x00\x01")
    (src / "empty.txt").write_bytes(b"")
    rep = corpus.ingest_paths(ws, [src, tmp_path / "missing.txt"], origin="test", rights=RIGHTS)
    codes = {i.code for i in rep.issues}
    assert {"corrupt_file", "unsupported_format", "empty_file", "not_found"} <= codes
    assert len(rep.duplicates) == 1 and len(rep.ingested) == 3
    m = ws.sources()
    for s in m.sources:
        raw = (ws.root / "sources" / "raw" / s.source_id / s.original_filename).read_bytes()
        assert sha256_bytes(raw) == s.sha256
    # re-running ingests nothing new and does not duplicate
    rep2 = corpus.ingest_paths(ws, [src / "a.txt"], origin="test", rights=RIGHTS)
    assert rep2.ingested == [] and len(rep2.duplicates) == 1
    insp = corpus.inspect_corpus(ws)
    assert insp["totals"]["reference"] == 1 and insp["reference_tables_flagged_for_retrieval"]


def test_terms_deterministic_with_provenance():
    rows = [{"source_id": "s", "chunk_id": f"c{i}", "text": "The hydraulic pump drives the hydraulic actuator. Pump pressure rises."} for i in range(3)]
    t1 = extract_terms(rows)
    assert t1 == extract_terms(list(reversed(rows)))
    top = {t["term"] for t in t1}
    assert "hydraulic" in top and "hydraulic pump" in top
    assert t1[0]["chunks"][0].startswith("s/c")


def test_pdf_extraction_and_needs_ocr_pages():
    pytest.importorskip("pypdf")
    raw = make_pdf([[f"Intro line number {i} about turbine blades and rotor balance." for i in range(2)], None,
                    ["Third page talks about bearings and lubrication schedules."]])
    ex = extract("m.pdf", raw)
    assert ex.pages_total == 3 and ex.needs_ocr_pages == [2]
    assert any(i.code == "needs_ocr" and i.page == 2 for i in ex.issues)
    assert {b.page for b in ex.blocks} == {1, 3}
    out = ingest_file(raw, filename="m.pdf", origin="o", rights=RIGHTS, ingested_on="2026-10-05")
    assert out.ok and out.stream and out.needs_ocr_pages == [2]
    assert out.chunks[0].record.page_start == 1


def test_pdf_fully_scanned_is_reported_not_ingested():
    pytest.importorskip("pypdf")
    out = ingest_file(make_pdf([None, None]), filename="s.pdf", origin="o", rights=RIGHTS, ingested_on="2026-10-05")
    assert not out.ok and any(i.code == "needs_ocr" for i in out.issues)


def test_pdf_missing_dependency_message(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "pypdf":
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    ex = extract("m.pdf", b"%PDF-1.4\n")
    assert ex.issues[0].code == "missing_dependency" and "pypdf" in ex.issues[0].detail


def test_cli_ingest_and_inspect(tmp_path, capsys):
    from llmtrainer.cli import main

    main(["init", str(tmp_path / "p"), "--name", "C", "--domain", "d"])
    (tmp_path / "a.txt").write_text(PROSE * 4)
    (tmp_path / "b.docx").write_bytes(b"junk")
    capsys.readouterr()
    assert main(["ingest", str(tmp_path / "p"), str(tmp_path / "a.txt"), str(tmp_path / "b.docx"), "--rights-status", "owned"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["ingested"]) == 1 and out["issues"][0]["code"] == "corrupt_file"
    assert main(["inspect-corpus", str(tmp_path / "p")]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["totals"]["train"] >= 1 and rep["top_terms"]
