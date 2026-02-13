"""
Document Parser Module for IBM/HCL TWS Documentation.

This module provides specialized parsers for ingesting IBM/HCL Workload Automation
documentation into the RAG system:

- PDF Parser: Extracts text from IBM TWS PDF manuals with table support
- HTML Parser: Parses IBM Knowledge Center and HCL documentation pages
- Markdown Parser: Processes TWS markdown documentation

Features:
- Structured extraction preserving document hierarchy
- Table extraction from PDFs
- Code block preservation from HTML
- Metadata extraction (version, product, section)
- Encoding handling for multi-language docs
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    JSON = "json"
    XML = "xml"


@dataclass
class DocumentChunk:
    """Represents a chunk of parsed document."""

    content: str
    chunk_index: int
    total_chunks: int
    source: str
    document_type: DocumentType
    metadata: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""

    def __post_init__(self):
        if not self.sha256:
            self.sha256 = hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass
class ParsedDocument:
    """Represents a fully parsed document."""

    title: str
    content: str
    source: str
    document_type: DocumentType
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[dict[str, str]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    parsed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def sha256(self) -> str:
        """Generate SHA256 hash of document content."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


# =============================================================================
# PDF Parser
# =============================================================================


class PDFParser:
    """
    Enhanced PDF parser for IBM/HCL TWS documentation.

    Features:
    - Text extraction with layout preservation
    - Table detection and extraction
    - Header/footer removal
    - Page number handling
    - Metadata extraction
    """

    def __init__(
        self,
        extract_tables: bool = True,
        remove_headers_footers: bool = True,
        preserve_layout: bool = True,
    ):
        """
        Initialize PDF parser.

        Args:
            extract_tables: Whether to extract tables separately
            remove_headers_footers: Whether to remove headers/footers
            preserve_layout: Whether to preserve text layout
        """
        self.extract_tables = extract_tables
        self.remove_headers_footers = remove_headers_footers
        self.preserve_layout = preserve_layout

        # Check for pypdf availability
        try:
            import pypdf  # noqa: F401

            self._pypdf_available = True
        except ImportError:
            self._pypdf_available = False
            logger.warning("pypdf not installed. PDF parsing will be limited.")

    def parse(self, source: str | Path | bytes | io.BytesIO) -> ParsedDocument:
        """
        Parse a PDF document.

        Args:
            source: File path, bytes, or BytesIO object

        Returns:
            ParsedDocument with extracted content
        """
        if not self._pypdf_available:
            raise ImportError("pypdf is required for PDF parsing. Install with: pip install pypdf")

        import pypdf  # noqa: F401

        # Handle different input types
        if isinstance(source, (str, Path)):
            source_path = str(source)
            reader = pypdf.PdfReader(source_path)
            source_name = Path(source).name
        elif isinstance(source, bytes):
            reader = pypdf.PdfReader(io.BytesIO(source))
            source_name = "uploaded.pdf"
        else:
            reader = pypdf.PdfReader(source)
            source_name = "stream.pdf"

        # Extract metadata
        metadata = self._extract_metadata(reader)

        # Extract text from all pages
        full_text = []
        sections = []
        tables = []

        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""

            if self.remove_headers_footers:
                page_text = self._remove_headers_footers(page_text)

            # Detect sections by heading patterns
            page_sections = self._detect_sections(page_text, page_num)
            sections.extend(page_sections)

            # Extract tables if enabled
            if self.extract_tables:
                page_tables = self._extract_tables_from_page(page_text, page_num)
                tables.extend(page_tables)

            full_text.append(page_text)

        content = "\n\n".join(full_text)

        # Clean up content
        content = self._clean_text(content)

        return ParsedDocument(
            title=metadata.get("title", source_name),
            content=content,
            source=source_name,
            document_type=DocumentType.PDF,
            metadata=metadata,
            sections=sections,
            tables=tables,
        )

    def _extract_metadata(self, reader) -> dict[str, Any]:
        """Extract metadata from PDF."""
        metadata = {}

        try:
            if reader.metadata:
                metadata["title"] = reader.metadata.get("/Title", "")
                metadata["author"] = reader.metadata.get("/Author", "")
                metadata["subject"] = reader.metadata.get("/Subject", "")
                metadata["creator"] = reader.metadata.get("/Creator", "")
                metadata["producer"] = reader.metadata.get("/Producer", "")
                metadata["creation_date"] = str(reader.metadata.get("/CreationDate", ""))

            metadata["num_pages"] = len(reader.pages)

            # Try to detect IBM/HCL product info
            if reader.pages:
                first_page = reader.pages[0].extract_text() or ""
                metadata.update(self._detect_tws_info(first_page))

        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {e}")

        return metadata

    def _detect_tws_info(self, text: str) -> dict[str, Any]:
        """Detect TWS/HCL product information from text."""
        info = {}

        # Version patterns
        version_patterns = [
            r"(?:Version|V)\s*(\d+\.\d+(?:\.\d+)?)",
            r"TWS\s+(\d+\.\d+(?:\.\d+)?)",
            r"Workload\s+Automation\s+(\d+\.\d+(?:\.\d+)?)",
        ]

        for pattern in version_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info["tws_version"] = match.group(1)
                break

        # Product patterns
        if "HCL Workload Automation" in text or "HCL TWS" in text:
            info["product"] = "HCL Workload Automation"
        elif "IBM Workload Automation" in text or "IBM TWS" in text:
            info["product"] = "IBM Workload Automation"
        elif "Tivoli Workload Scheduler" in text:
            info["product"] = "IBM Tivoli Workload Scheduler"

        return info

    def _remove_headers_footers(self, text: str) -> str:
        """Remove common header/footer patterns."""
        lines = text.split("\n")

        # Remove page numbers and common footer patterns
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()

            # Skip pure page numbers
            if re.match(r"^\d+$", stripped):
                continue

            # Skip common footer patterns
            if re.match(r"^Page\s+\d+\s+of\s+\d+", stripped, re.IGNORECASE):
                continue

            # Skip IBM/HCL copyright footers
            if re.match(r"^©\s*(IBM|HCL)\s+Corporation", stripped):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _detect_sections(self, text: str, page_num: int) -> list[dict[str, str]]:
        """Detect document sections by heading patterns."""
        sections = []

        # Common heading patterns for TWS docs
        heading_patterns = [
            r"^(\d+(?:\.\d+)*)\s+(.+)$",  # Numbered sections: 1.2.3 Title
            r"^(Chapter\s+\d+)[.:]\s*(.+)$",  # Chapter N: Title
            r"^([A-Z][A-Z\s]+)$",  # ALL CAPS headings
        ]

        for line in text.split("\n"):
            stripped = line.strip()
            for pattern in heading_patterns:
                match = re.match(pattern, stripped)
                if match:
                    sections.append(
                        {
                            "level": match.group(1) if len(match.groups()) > 1 else "1",
                            "title": match.group(2) if len(match.groups()) > 1 else match.group(1),
                            "page": page_num,
                        }
                    )
                    break

        return sections

    def _extract_tables_from_page(self, text: str, page_num: int) -> list[dict[str, Any]]:
        """Extract table-like structures from page text."""
        tables = []

        # Simple table detection based on consistent spacing/alignment
        lines = text.split("\n")
        table_lines = []
        in_table = False

        for line in lines:
            # Detect table rows by multiple tab/space separations
            if re.search(r"\s{2,}|\t", line) and len(line.split()) >= 2:
                table_lines.append(line)
                in_table = True
            elif in_table and line.strip():
                # Check if still in table
                if re.search(r"\s{2,}|\t", line):
                    table_lines.append(line)
                else:
                    # End of table
                    if len(table_lines) >= 3:  # Minimum 3 rows for a table
                        tables.append(
                            {
                                "page": page_num,
                                "rows": len(table_lines),
                                "content": "\n".join(table_lines),
                            }
                        )
                    table_lines = []
                    in_table = False

        # Handle table at end of page
        if len(table_lines) >= 3:
            tables.append(
                {
                    "page": page_num,
                    "rows": len(table_lines),
                    "content": "\n".join(table_lines),
                }
            )

        return tables

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Fix common OCR issues
        text = text.replace("ﬁ", "fi")
        text = text.replace("ﬂ", "fl")
        text = text.replace("'", "'")
        text = text.replace(
            """, '"')
        text = text.replace(""",
            '"',
        )

        return text.strip()


# =============================================================================
# HTML Parser
# =============================================================================


class HTMLParser:
    """
    HTML parser for IBM Knowledge Center and HCL documentation.

    Features:
    - BeautifulSoup4-based parsing
    - Code block preservation
    - Table extraction
    - Navigation/sidebar removal
    - Link resolution
    """

    def __init__(
        self,
        remove_navigation: bool = True,
        preserve_code_blocks: bool = True,
        extract_links: bool = True,
    ):
        """
        Initialize HTML parser.

        Args:
            remove_navigation: Whether to remove nav/sidebar elements
            preserve_code_blocks: Whether to preserve code block formatting
            extract_links: Whether to extract and store links
        """
        self.remove_navigation = remove_navigation
        self.preserve_code_blocks = preserve_code_blocks
        self.extract_links = extract_links

        # Check for BeautifulSoup availability
        try:
            from bs4 import BeautifulSoup  # noqa: F401

            self._bs4_available = True
        except ImportError:
            self._bs4_available = False
            logger.warning("BeautifulSoup4 not installed. HTML parsing will be limited.")

    def parse(
        self,
        source: str | bytes,
        url: str | None = None,
    ) -> ParsedDocument:
        """
        Parse an HTML document.

        Args:
            source: HTML content as string or bytes, or file path
            url: Optional source URL for link resolution

        Returns:
            ParsedDocument with extracted content
        """
        if not self._bs4_available:
            raise ImportError(
                "BeautifulSoup4 is required for HTML parsing. "
                "Install with: pip install beautifulsoup4"
            )

        from bs4 import BeautifulSoup  # noqa: F401

        # Handle file path
        if isinstance(source, str) and os.path.exists(source):
            with open(source, encoding="utf-8") as f:
                html_content = f.read()
            source_name = Path(source).name
        elif isinstance(source, bytes):
            html_content = source.decode("utf-8", errors="replace")
            source_name = url or "uploaded.html"
        else:
            html_content = source
            source_name = url or "content.html"

        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract metadata
        metadata = self._extract_metadata(soup, url)

        # Remove unwanted elements
        if self.remove_navigation:
            self._remove_navigation_elements(soup)

        # Extract main content
        main_content = self._extract_main_content(soup)

        # Extract sections
        sections = self._extract_sections(soup)

        # Extract tables
        tables = self._extract_tables(soup)

        # Extract links
        if self.extract_links:
            metadata["links"] = self._extract_links(soup, url)

        # Convert to text
        content = self._soup_to_text(main_content or soup)

        return ParsedDocument(
            title=metadata.get("title", source_name),
            content=content,
            source=source_name,
            document_type=DocumentType.HTML,
            metadata=metadata,
            sections=sections,
            tables=tables,
        )

    def _extract_metadata(self, soup, url: str | None) -> dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {}

        # Title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

        # Meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name", "").lower()
            content = meta.get("content", "")

            if name in ["description", "keywords", "author", "version"]:
                metadata[name] = content
            elif name == "dc.date" or name == "date":
                metadata["date"] = content

        # URL
        if url:
            metadata["url"] = url
            metadata["domain"] = urlparse(url).netloc

        # Detect IBM/HCL content
        page_text = soup.get_text()[:1000]  # First 1000 chars
        if "IBM Knowledge Center" in page_text:
            metadata["source_type"] = "ibm_knowledge_center"
        elif "HCL Documentation" in page_text or "HCL Software" in page_text:
            metadata["source_type"] = "hcl_documentation"

        return metadata

    def _remove_navigation_elements(self, soup) -> None:
        """Remove navigation, sidebar, and header/footer elements."""
        # Common navigation selectors
        selectors = [
            "nav",
            "header",
            "footer",
            ".navigation",
            ".nav",
            ".navbar",
            ".sidebar",
            ".side-nav",
            ".sidenav",
            ".toc",
            ".table-of-contents",
            ".breadcrumb",
            ".breadcrumbs",
            "#header",
            "#footer",
            "#nav",
            "#sidebar",
            "[role='navigation']",
            "[role='banner']",
            ".ibm-masthead",
            ".ibm-footer",  # IBM-specific
            ".hcl-header",
            ".hcl-footer",  # HCL-specific
        ]

        for selector in selectors:
            for element in soup.select(selector):
                element.decompose()

    def _extract_main_content(self, soup):
        """Extract main content area."""
        # Common main content selectors
        selectors = [
            "main",
            "article",
            ".content",
            ".main-content",
            ".doc-content",
            "#content",
            "#main-content",
            "#article",
            ".ibm-content",  # IBM-specific
            ".hcl-content",  # HCL-specific
            "[role='main']",
        ]

        for selector in selectors:
            main = soup.select_one(selector)
            if main:
                return main

        # Fallback to body
        return soup.find("body") or soup

    def _extract_sections(self, soup) -> list[dict[str, str]]:
        """Extract document sections by headings."""
        sections = []

        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                section_id = heading.get("id", "")
                sections.append(
                    {
                        "level": str(level),
                        "title": heading.get_text(strip=True),
                        "id": section_id,
                    }
                )

        return sections

    def _extract_tables(self, soup) -> list[dict[str, Any]]:
        """Extract tables from HTML."""
        tables = []

        for idx, table in enumerate(soup.find_all("table")):
            table_data = {
                "index": idx,
                "headers": [],
                "rows": [],
            }

            # Extract headers
            thead = table.find("thead")
            if thead:
                for th in thead.find_all("th"):
                    table_data["headers"].append(th.get_text(strip=True))

            # Extract rows
            tbody = table.find("tbody") or table
            for tr in tbody.find_all("tr"):
                row = []
                for td in tr.find_all(["td", "th"]):
                    row.append(td.get_text(strip=True))
                if row:
                    table_data["rows"].append(row)

            if table_data["rows"]:
                tables.append(table_data)

        return tables

    def _extract_links(self, soup, base_url: str | None) -> list[dict[str, str]]:
        """Extract and resolve links."""
        links = []

        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            text = a.get_text(strip=True)

            # Skip empty or anchor-only links
            if not href or href.startswith("#"):
                continue

            # Resolve relative URLs
            if base_url and not href.startswith(("http://", "https://", "//")):
                href = urljoin(base_url, href)

            links.append(
                {
                    "url": href,
                    "text": text,
                }
            )

        return links

    def _soup_to_text(self, soup) -> str:
        """Convert BeautifulSoup element to clean text."""
        # Handle code blocks specially
        if self.preserve_code_blocks:
            for code in soup.find_all(["code", "pre"]):
                code_text = code.get_text()
                code.replace_with(f"\n```\n{code_text}\n```\n")

        # Get text with separator
        text = soup.get_text(separator="\n")

        # Clean up
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()


# =============================================================================
# URL Fetcher for Online Documentation
# =============================================================================


class DocumentationFetcher:
    """
    Fetch documentation from IBM Knowledge Center and HCL Documentation.

    Handles:
    - HTTP requests with proper headers
    - Rate limiting
    - Error handling
    - Caching
    """

    # Common documentation base URLs
    IBM_KC_BASE = "https://www.ibm.com/docs"
    HCL_DOCS_BASE = "https://help.hcltechsw.com"

    def __init__(
        self,
        timeout: float = 30.0,
        user_agent: str | None = None,
        cache_dir: Path | None = None,
    ):
        """
        Initialize documentation fetcher.

        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
            cache_dir: Directory for caching fetched pages
        """
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (compatible; ResyncRAG/1.0; +https://github.com/resync)"
        )
        self.cache_dir = cache_dir

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch(self, url: str) -> tuple[str, dict[str, Any]]:
        """
        Fetch a documentation page.

        Args:
            url: URL to fetch

        Returns:
            Tuple of (content, metadata)
        """
        import httpx

        # Check cache first
        if self.cache_dir:
            cache_file = self._get_cache_path(url)
            if cache_file.exists():
                logger.debug(f"Using cached content for {url}")
                return cache_file.read_text(encoding="utf-8"), {"cached": True}

        # Fetch page
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            content = response.text

        # Cache the content
        if self.cache_dir:
            cache_file = self._get_cache_path(url)
            cache_file.write_text(content, encoding="utf-8")

        metadata = {
            "url": str(response.url),
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        return content, metadata

    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for URL."""
        url_hash = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()
        return self.cache_dir / f"{url_hash}.html"


# =============================================================================
# Unified Document Parser
# =============================================================================


class DocumentParser:
    """
    Unified document parser for RAG ingestion.

    Automatically detects document type and uses appropriate parser.
    """

    def __init__(self):
        """Initialize document parser with all sub-parsers."""
        self.pdf_parser = PDFParser()
        self.html_parser = HTMLParser()

    def parse(
        self,
        source: str | Path | bytes,
        document_type: DocumentType | None = None,
        **kwargs,
    ) -> ParsedDocument:
        """
        Parse a document.

        Args:
            source: Document source (path, bytes, or string)
            document_type: Explicit document type (auto-detected if not provided)
            **kwargs: Additional parser-specific options

        Returns:
            ParsedDocument
        """
        # Auto-detect document type
        if document_type is None:
            document_type = self._detect_type(source)

        # Route to appropriate parser
        if document_type == DocumentType.PDF:
            return self.pdf_parser.parse(source)
        if document_type == DocumentType.HTML:
            return self.html_parser.parse(source, **kwargs)
        if document_type == DocumentType.MARKDOWN:
            return self._parse_markdown(source)
        if document_type == DocumentType.TEXT:
            return self._parse_text(source)
        raise ValueError(f"Unsupported document type: {document_type}")

    def _detect_type(self, source: str | Path | bytes) -> DocumentType:
        """Auto-detect document type."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                suffix = path.suffix.lower()
                type_map = {
                    ".pdf": DocumentType.PDF,
                    ".html": DocumentType.HTML,
                    ".htm": DocumentType.HTML,
                    ".md": DocumentType.MARKDOWN,
                    ".txt": DocumentType.TEXT,
                    ".json": DocumentType.JSON,
                    ".xml": DocumentType.XML,
                }
                return type_map.get(suffix, DocumentType.TEXT)
            if source.startswith(("http://", "https://")) or source.strip().startswith(
                ("<html", "<!DOCTYPE", "<HTML")
            ):
                return DocumentType.HTML
        elif isinstance(source, bytes):
            if source.startswith(b"%PDF"):
                return DocumentType.PDF
            if source.startswith((b"<html", b"<!DOCTYPE", b"<HTML")):
                return DocumentType.HTML

        return DocumentType.TEXT

    def _parse_markdown(self, source: str | Path) -> ParsedDocument:
        """Parse markdown document."""
        if isinstance(source, Path) or os.path.exists(str(source)):
            path = Path(source)
            content = path.read_text(encoding="utf-8")
            source_name = path.name
        else:
            content = source
            source_name = "content.md"

        # Extract title from first heading
        title_match = re.match(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else source_name

        # Extract sections
        sections = []
        for match in re.finditer(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE):
            sections.append(
                {
                    "level": str(len(match.group(1))),
                    "title": match.group(2),
                }
            )

        return ParsedDocument(
            title=title,
            content=content,
            source=source_name,
            document_type=DocumentType.MARKDOWN,
            sections=sections,
        )

    def _parse_text(self, source: str | Path) -> ParsedDocument:
        """Parse plain text document."""
        if isinstance(source, Path) or os.path.exists(str(source)):
            path = Path(source)
            content = path.read_text(encoding="utf-8")
            source_name = path.name
        else:
            content = source
            source_name = "content.txt"

        return ParsedDocument(
            title=source_name,
            content=content,
            source=source_name,
            document_type=DocumentType.TEXT,
        )

    def chunk(
        self,
        document: ParsedDocument,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Generator[DocumentChunk, None, None]:
        """
        Chunk a parsed document.

        Args:
            document: ParsedDocument to chunk
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks

        Yields:
            DocumentChunk objects
        """
        content = document.content
        total_length = len(content)

        if total_length <= chunk_size:
            yield DocumentChunk(
                content=content,
                chunk_index=0,
                total_chunks=1,
                source=document.source,
                document_type=document.document_type,
                metadata=document.metadata,
            )
            return

        # Calculate chunks
        chunks = []
        start = 0

        while start < total_length:
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < total_length:
                # Look for sentence end
                for sep in [". ", ".\n", "\n\n", "\n", " "]:
                    break_point = content.rfind(sep, start + chunk_size // 2, end)
                    if break_point != -1:
                        end = break_point + len(sep)
                        break

            chunks.append(content[start:end])
            start = end - chunk_overlap

        total_chunks = len(chunks)
        for idx, chunk_content in enumerate(chunks):
            yield DocumentChunk(
                content=chunk_content,
                chunk_index=idx,
                total_chunks=total_chunks,
                source=document.source,
                document_type=document.document_type,
                metadata={
                    **document.metadata,
                    "title": document.title,
                },
            )


# =============================================================================
# Factory Functions
# =============================================================================


def create_pdf_parser(**kwargs) -> PDFParser:
    """Create a PDF parser instance."""
    return PDFParser(**kwargs)


def create_html_parser(**kwargs) -> HTMLParser:
    """Create an HTML parser instance."""
    return HTMLParser(**kwargs)


def create_document_parser() -> DocumentParser:
    """Create a unified document parser instance."""
    return DocumentParser()
