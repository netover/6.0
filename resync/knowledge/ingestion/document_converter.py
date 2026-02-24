# pylint
# mypy
"""
Document Converter Service using Docling.

Converts PDF, DOCX, HTML, Markdown, and XLSX files into structured text
suitable for RAG ingestion. Runs Docling in an isolated subprocess to
contain its ~3-4 GB memory footprint.

Architecture:
    Upload → DoclingConverter.convert() → ConvertedDocument → IngestService

Usage:
    converter = DoclingConverter()
    result = await converter.convert("/path/to/manual.pdf")
    # result.markdown  → full document as markdown
    # result.tables    → list of extracted tables (as dicts)
    # result.metadata  → title, pages, format, sections
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SupportedFormat(str, Enum):
    """File formats supported by the converter."""

    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "md"
    XLSX = "xlsx"
    TXT = "txt"


FORMAT_EXTENSIONS: dict[str, SupportedFormat] = {
    ".pdf": SupportedFormat.PDF,
    ".docx": SupportedFormat.DOCX,
    ".doc": SupportedFormat.DOCX,
    ".html": SupportedFormat.HTML,
    ".htm": SupportedFormat.HTML,
    ".md": SupportedFormat.MARKDOWN,
    ".markdown": SupportedFormat.MARKDOWN,
    ".xlsx": SupportedFormat.XLSX,
    ".txt": SupportedFormat.TXT,
    ".text": SupportedFormat.TXT,
}


@dataclass
class ExtractedTable:
    """A table extracted from a document."""

    content_markdown: str
    page: int | None = None
    section: str | None = None
    rows: int = 0
    cols: int = 0


@dataclass
class ConvertedDocument:
    """Result of a document conversion."""

    markdown: str
    tables: list[ExtractedTable] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""
    format: str = ""
    pages: int = 0
    conversion_time_s: float = 0.0
    status: str = "success"
    error: str | None = None


def _docling_convert_worker(
    file_path: str,
    do_table_structure: bool,
    do_ocr: bool,
    doc_timeout: int,
    result_path: str,
) -> None:
    """
    Run Docling conversion in a subprocess.

    This function is the target of multiprocessing.Process. It imports
    Docling only inside the subprocess, so the main process never pays
    the ~3-4 GB memory cost.

    Results are written to a temp JSON file to avoid pickling large objects.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_table_structure = do_table_structure
        pdf_opts.do_ocr = do_ocr
        converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.MD,
                InputFormat.XLSX,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
            },
        )
        result = converter.convert(file_path)
        doc = result.document
        markdown = doc.export_to_markdown()
        tables = []
        for table_item in doc.tables:
            try:
                md_table = table_item.export_to_markdown()
                df = table_item.export_to_dataframe()
                rows, cols = df.shape
            except Exception:
                md_table = str(table_item)
                rows, cols = (0, 0)
            tables.append(
                {
                    "content_markdown": md_table,
                    "page": getattr(table_item.prov[0], "page_no", None)
                    if table_item.prov
                    else None,
                    "rows": rows,
                    "cols": cols,
                }
            )
        metadata = {
            "title": getattr(doc, "name", "") or Path(file_path).stem,
            "num_pages": getattr(result.document, "num_pages", 0)
            if hasattr(result.document, "num_pages")
            else 0,
            "num_tables": len(tables),
        }
        output = {
            "status": "success",
            "markdown": markdown,
            "tables": tables,
            "metadata": metadata,
        }
    except Exception as e:
        output = {
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
            "markdown": "",
            "tables": [],
            "metadata": {},
        }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, default=str)


def _plaintext_fallback(file_path: str) -> dict:
    """Fallback: read file as plain text when Docling is unavailable."""
    try:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return {
            "status": "success",
            "markdown": text,
            "tables": [],
            "metadata": {"title": Path(file_path).stem, "fallback": True},
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "markdown": "",
            "tables": [],
            "metadata": {},
        }


class DoclingConverter:
    """
    Document converter using Docling in an isolated subprocess.

    The subprocess is spawned on-demand per conversion and killed after
    completion, so Docling's ~3-4 GB memory is fully reclaimed.

    Usage:
        converter = DoclingConverter()
        result = await converter.convert("manual.pdf")
        print(result.markdown[:200])
        print(f"Extracted {len(result.tables)} tables")
    """

    def __init__(
        self,
        *,
        do_table_structure: bool = True,
        do_ocr: bool = False,
        doc_timeout: int = 120,
        process_timeout: int = 300,
    ):
        """
        Args:
            do_table_structure: Use ML model for table extraction (recommended).
            do_ocr: Enable OCR for scanned documents (slower, more RAM).
            doc_timeout: Docling internal timeout per document (seconds).
            process_timeout: Hard timeout for the subprocess (seconds).
        """
        self.do_table_structure = do_table_structure
        self.do_ocr = do_ocr
        self.doc_timeout = doc_timeout
        self.process_timeout = process_timeout

    @staticmethod
    def detect_format(file_path: str | Path) -> SupportedFormat | None:
        """Detect document format from file extension."""
        ext = Path(file_path).suffix.lower()
        return FORMAT_EXTENSIONS.get(ext)

    @staticmethod
    def is_supported(file_path: str | Path) -> bool:
        """Check if a file format is supported."""
        return DoclingConverter.detect_format(file_path) is not None

    async def convert(self, file_path: str | Path) -> ConvertedDocument:
        """
        Convert a document to structured markdown with extracted tables.

        Runs Docling in a subprocess to isolate memory usage (~4 GB).
        If the subprocess crashes on a specific file, falls back to
        plain text extraction for that file only.

        Args:
            file_path: Path to the document file.

        Returns:
            ConvertedDocument with markdown, tables, and metadata.
        """
        file_path = Path(file_path).resolve()
        fmt = self.detect_format(file_path)
        if not file_path.exists():
            return ConvertedDocument(
                markdown="",
                source_path=str(file_path),
                status="error",
                error=f"File not found: {file_path}",
            )
        if fmt is None:
            return ConvertedDocument(
                markdown="",
                source_path=str(file_path),
                status="error",
                error=f"Unsupported format: {file_path.suffix}",
            )
        if fmt == SupportedFormat.TXT:
            data = _plaintext_fallback(str(file_path))
            return self._build_result(data, str(file_path), fmt.value)
        t0 = time.perf_counter()
        logger.info(
            "docling_convert_start", extra={"file": str(file_path), "format": fmt.value}
        )
        data = await self._run_in_subprocess(str(file_path))
        elapsed = time.perf_counter() - t0
        result = self._build_result(data, str(file_path), fmt.value)
        result.conversion_time_s = elapsed
        if result.status == "success":
            logger.info(
                "docling_convert_done",
                extra={
                    "file": file_path.name,
                    "pages": result.pages,
                    "tables": len(result.tables),
                    "time_s": f"{elapsed:.1f}",
                },
            )
        else:
            logger.warning(
                "docling_convert_failed",
                extra={
                    "file": file_path.name,
                    "error": result.error,
                    "time_s": f"{elapsed:.1f}",
                },
            )
        return result

    async def convert_batch(
        self, file_paths: list[str | Path], max_concurrent: int = 2
    ) -> list[ConvertedDocument]:
        """
        Convert multiple documents with limited concurrency.

        Args:
            file_paths: List of file paths to convert.
            max_concurrent: Maximum concurrent conversions (each uses ~4 GB RAM).

        Returns:
            List of ConvertedDocument results (same order as input).
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited(fp: str | Path) -> ConvertedDocument:
            async with semaphore:
                return await self.convert(fp)

        return await asyncio.gather(*[_limited(fp) for fp in file_paths])

    async def _run_in_subprocess(self, file_path: str) -> dict:
        """Spawn Docling in a subprocess and collect results."""
        fd, result_path = tempfile.mkstemp(suffix=".json", prefix="docling_")
        os.close(fd)
        try:
            loop = asyncio.get_running_loop()
            result_data = await loop.run_in_executor(
                None, self._sync_subprocess, file_path, result_path
            )
            return result_data
        finally:
            try:
                os.unlink(result_path)
            except OSError:
                pass

    def _sync_subprocess(self, file_path: str, result_path: str) -> dict:
        """Synchronous subprocess execution (called from executor)."""
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_docling_convert_worker,
            args=(
                file_path,
                self.do_table_structure,
                self.do_ocr,
                self.doc_timeout,
                result_path,
            ),
            daemon=True,
        )
        proc.start()
        proc.join(timeout=self.process_timeout)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
            return {
                "status": "error",
                "error": f"Conversion timed out after {self.process_timeout}s",
                "markdown": "",
                "tables": [],
                "metadata": {},
            }
        if proc.exitcode != 0:
            logger.warning(
                "docling_subprocess_crashed",
                extra={"exitcode": proc.exitcode, "file": file_path},
            )
            return _plaintext_fallback(file_path)
        try:
            with open(result_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return {
                "status": "error",
                "error": f"Failed to read conversion results: {e}",
                "markdown": "",
                "tables": [],
                "metadata": {},
            }

    @staticmethod
    def _build_result(data: dict, source_path: str, fmt: str) -> ConvertedDocument:
        """Build ConvertedDocument from subprocess output."""
        tables = [
            ExtractedTable(
                content_markdown=t.get("content_markdown", ""),
                page=t.get("page"),
                section=t.get("section"),
                rows=t.get("rows", 0),
                cols=t.get("cols", 0),
            )
            for t in data.get("tables", [])
        ]
        metadata = data.get("metadata", {})
        return ConvertedDocument(
            markdown=data.get("markdown", ""),
            tables=tables,
            metadata=metadata,
            source_path=source_path,
            format=fmt,
            pages=metadata.get("num_pages", 0),
            status=data.get("status", "error"),
            error=data.get("error"),
        )


async def convert_document(
    file_path: str | Path, *, do_table_structure: bool = True, do_ocr: bool = False
) -> ConvertedDocument:
    """
    One-shot document conversion.

    Args:
        file_path: Path to the document.
        do_table_structure: Use ML table extraction.
        do_ocr: Enable OCR for scanned docs.

    Returns:
        ConvertedDocument with markdown and tables.
    """
    converter = DoclingConverter(do_table_structure=do_table_structure, do_ocr=do_ocr)
    return await converter.convert(file_path)
