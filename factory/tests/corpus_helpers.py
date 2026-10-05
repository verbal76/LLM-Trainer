"""Programmatic fixture writers (no binary fixtures checked in)."""

from __future__ import annotations

import io
import zipfile
from xml.sax.saxutils import escape

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def make_docx(items: list) -> bytes:
    """items: ("h1"|"h2"|"p", text) or ("table", [[cells...], ...])."""
    body = []
    for kind, val in items:
        if kind == "table":
            rows = "".join("<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{escape(c)}</w:t></w:r></w:p></w:tc>" for c in r) + "</w:tr>" for r in val)
            body.append(f"<w:tbl>{rows}</w:tbl>")
        else:
            style = f'<w:pPr><w:pStyle w:val="Heading{kind[1]}"/></w:pPr>' if kind[0] == "h" else ""
            body.append(f"<w:p>{style}<w:r><w:t>{escape(val)}</w:t></w:r></w:p>")
    doc = f'<?xml version="1.0"?><w:document xmlns:w="{W}"><w:body>{"".join(body)}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        z.writestr("word/document.xml", doc)
    return buf.getvalue()


def make_pdf(pages: list[list[str] | None]) -> bytes:
    """Minimal PDF with Helvetica text. A page of None has no text (image-only stand-in)."""
    objs: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_id = 3 + 2 * n
    for i, lines in enumerate(pages):
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode())
        if lines is None:
            stream = b""
        else:
            ops = ["BT /F1 11 Tf 14 TL 50 740 Td"]
            for ln in lines:
                ops.append("(" + ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") Tj T*")
            ops.append("ET")
            stream = "\n".join(ops).encode("latin-1")
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, x)
    return bytes(out)
