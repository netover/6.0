"""
Advanced Chunking System for Resync RAG.

Implements state-of-the-art chunking strategies for technical documentation:
- Structure-aware parsing (markdown headers, code blocks, tables)
- Semantic chunking using sentence transformers
- Contextual enrichment (parent headers, summaries)
- Hierarchical multi-level chunking
- TWS-specific optimizations (error codes, job names, procedures)

Based on research from Anthropic (Contextual Retrieval), Jina (Late Chunking),
NVIDIA, and LangChain best practices.

Key improvements over basic chunking:
- 35-67% reduction in retrieval failures
- Preserves atomic units (error codes + descriptions together)
- Multi-granularity for different query types

Author: Resync Team
Version: 5.4.2
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False
    _ENC = None

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None
    np = None


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"  # Basic token-based
    RECURSIVE = "recursive"  # Hierarchical separators
    SEMANTIC = "semantic"  # Embedding similarity
    STRUCTURE_AWARE = "structure_aware"  # Markdown/document structure
    HIERARCHICAL = "hierarchical"  # Multi-level granularity
    TWS_OPTIMIZED = "tws_optimized"  # TWS-specific patterns


class ChunkType(str, Enum):
    """Type of content in a chunk."""

    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    ERROR_DOC = "error_documentation"
    PROCEDURE = "procedure"
    HEADER = "header"
    DEFINITION = "definition"


# TWS-specific patterns
TWS_ERROR_PATTERN = re.compile(
    r"(AWS[A-Z]{2,4}\d{3}[EWI]?|AWSJ[A-Z]+\d+[EWI]?|"
    r"AWKR[A-Z]+\d+[EWI]?|JAW[A-Z]+\d+[EWI]?)",
    re.IGNORECASE,
)
TWS_JOB_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{4,15}\b")
TWS_COMMAND_PATTERN = re.compile(
    r"\b(conman|optman|composer|planman|stageman|"
    r"datecalc|jnextday|batchman|mailman|jobman)\b",
    re.IGNORECASE,
)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ChunkMetadata:
    """Metadata for a chunk with authority and freshness signals."""

    # Source info
    source_file: str = ""
    document_title: str = ""

    # Position
    chunk_index: int = 0
    total_chunks: int = 0
    start_char: int = 0
    end_char: int = 0

    # Hierarchy
    parent_headers: list[str] = field(default_factory=list)
    section_path: str = ""  # e.g., "Chapter 5 > Troubleshooting > Error Codes"

    # Content type
    chunk_type: ChunkType = ChunkType.TEXT

    # TWS-specific
    error_codes: list[str] = field(default_factory=list)
    job_names: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)

    # Context
    prev_chunk_summary: str = ""
    next_chunk_summary: str = ""

    # Quality
    token_count: int = 0
    semantic_coherence: float = 0.0

    # === PR1: Authority and Freshness Signals ===
    # Document type for authority scoring
    doc_type: str = "unknown"  # policy, manual, kb, blog, forum
    source_tier: str = "unknown"  # verified, official, curated, community, generated
    authority_tier: int = 3  # 1-5 (lower = more authoritative)

    # Freshness tracking
    doc_version: int = 1
    last_updated: str = ""  # ISO timestamp
    is_deprecated: bool = False

    # Embedding metadata
    embedding_model: str = ""
    embedding_version: str = ""

    # Platform/Environment for filtering
    platform: str = "all"  # ios, android, mobile, web, desktop, all
    environment: str = "all"  # prod, staging, dev, all

    # === v5.9.2: Provenance Tracking ===
    # Extraction provenance
    extraction_model: str = ""  # e.g., "gpt-4o", "llama-3.1-70b"
    extraction_timestamp: str = ""  # ISO timestamp of extraction
    extraction_method: str = "hybrid"  # llm, regex, hybrid, manual

    # Quality metrics
    confidence_score: float = 0.0  # 0.0-1.0 extraction confidence
    validation_passed: bool = False  # Ontology validation result
    validation_errors: list[str] = field(default_factory=list)

    # Human verification
    verified: bool = False  # Human reviewed?
    verified_by: str | None = None  # Who verified
    verified_at: str | None = None  # When verified (ISO timestamp)

    # Entity extraction results (from ontology-driven extraction)
    extracted_entities: list[dict[str, Any]] = field(default_factory=list)
    entity_count: int = 0  # Number of entities extracted from this chunk

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "source_file": self.source_file,
            "document_title": self.document_title,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "parent_headers": self.parent_headers,
            "section_path": self.section_path,
            "chunk_type": self.chunk_type.value,
            "error_codes": self.error_codes,
            "job_names": self.job_names,
            "commands": self.commands,
            "token_count": self.token_count,
            # Authority signals
            "doc_type": self.doc_type,
            "source_tier": self.source_tier,
            "authority_tier": self.authority_tier,
            # Freshness signals
            "doc_version": self.doc_version,
            "last_updated": self.last_updated,
            "is_deprecated": self.is_deprecated,
            # Embedding tracking
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            # Filtering
            "platform": self.platform,
            "environment": self.environment,
            # v5.9.2: Provenance tracking
            "extraction_model": self.extraction_model,
            "extraction_timestamp": self.extraction_timestamp,
            "extraction_method": self.extraction_method,
            "confidence_score": self.confidence_score,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "verified": self.verified,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at,
            "extracted_entities": self.extracted_entities,
            "entity_count": self.entity_count,
        }


@dataclass
class EnrichedChunk:
    """A chunk with content and rich metadata."""

    content: str
    contextualized_content: str  # Content with prepended context
    metadata: ChunkMetadata
    sha256: str = ""

    def __post_init__(self):
        if not self.sha256:
            self.sha256 = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "contextualized_content": self.contextualized_content,
            "sha256": self.sha256,
            "metadata": self.metadata.to_dict(),
        }


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""

    # Strategy
    strategy: ChunkingStrategy = ChunkingStrategy.TWS_OPTIMIZED

    # Size parameters
    max_tokens: int = 500
    min_tokens: int = 50
    overlap_tokens: int = 75  # 15% overlap

    # Semantic chunking
    semantic_threshold_percentile: float = 90.0
    semantic_buffer_size: int = 2
    embedding_model: str = "all-MiniLM-L6-v2"

    # Structure
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    preserve_error_docs: bool = True

    # Context enrichment
    add_parent_headers: bool = True
    add_section_path: bool = True
    generate_summaries: bool = False  # Requires LLM

    # Hierarchical
    hierarchy_levels: list[int] = field(default_factory=lambda: [2048, 512, 128])

    # TWS-specific
    extract_error_codes: bool = True
    extract_job_names: bool = True
    extract_commands: bool = True


# =============================================================================
# TOKEN UTILITIES
# =============================================================================


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    if _HAS_TIKTOKEN and _ENC:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to max tokens."""
    if _HAS_TIKTOKEN and _ENC:
        tokens = _ENC.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return _ENC.decode(tokens[:max_tokens])
    # Fallback: estimate 4 chars per token
    max_chars = max_tokens * 4
    return text[:max_chars]


# =============================================================================
# TEXT SPLITTING UTILITIES
# =============================================================================


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences with improved handling.

    Handles:
    - Standard punctuation
    - Abbreviations (e.g., "Dr.", "vs.")
    - Numbers with decimals
    - Error codes (e.g., "AWS001E")
    """
    if not text:
        return []

    # Protect abbreviations and numbers
    protected = text
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)  # Decimal numbers
    protected = re.sub(r"\b(Mr|Mrs|Dr|vs|etc|e\.g|i\.e)\.", r"\1<DOT>", protected)

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)

    # Restore protected dots
    sentences = [s.replace("<DOT>", ".") for s in sentences]

    # Clean up
    return [s.strip() for s in sentences if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


# =============================================================================
# MARKDOWN STRUCTURE DETECTION
# =============================================================================


@dataclass
class MarkdownSection:
    """A section of markdown document."""

    level: int  # 1-6 for # to ######
    title: str
    content: str
    start_line: int
    end_line: int
    children: list[MarkdownSection] = field(default_factory=list)


def parse_markdown_structure(text: str) -> list[MarkdownSection]:
    """
    Parse markdown into hierarchical sections.

    Preserves:
    - Header hierarchy
    - Code blocks (not split)
    - Tables (not split)
    """
    lines = text.split("\n")
    sections: list[MarkdownSection] = []
    current_section: MarkdownSection | None = None
    content_buffer: list[str] = []
    in_code_block = False

    for i, line in enumerate(lines):
        # Track code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            content_buffer.append(line)
            continue

        if in_code_block:
            content_buffer.append(line)
            continue

        # Check for header
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)

        if header_match:
            # Save previous section
            if current_section:
                current_section.content = "\n".join(content_buffer).strip()
                current_section.end_line = i - 1
                sections.append(current_section)

            # Start new section
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            current_section = MarkdownSection(
                level=level,
                title=title,
                content="",
                start_line=i,
                end_line=i,
            )
            content_buffer = []
        else:
            content_buffer.append(line)

    # Save last section
    if current_section:
        current_section.content = "\n".join(content_buffer).strip()
        current_section.end_line = len(lines) - 1
        sections.append(current_section)
    elif content_buffer:
        # Content without headers
        sections.append(
            MarkdownSection(
                level=0,
                title="",
                content="\n".join(content_buffer).strip(),
                start_line=0,
                end_line=len(lines) - 1,
            )
        )

    return sections


def build_section_hierarchy(sections: list[MarkdownSection]) -> list[MarkdownSection]:
    """Build hierarchical structure from flat sections."""
    if not sections:
        return []

    root_sections: list[MarkdownSection] = []
    stack: list[MarkdownSection] = []

    for section in sections:
        # Pop sections with same or higher level
        while stack and stack[-1].level >= section.level:
            stack.pop()

        if stack:
            stack[-1].children.append(section)
        else:
            root_sections.append(section)

        stack.append(section)

    return root_sections


def get_section_path(section: MarkdownSection, ancestors: list[str]) -> str:
    """Get full path of section (e.g., 'Chapter 5 > Troubleshooting > Errors')."""
    path_parts = ancestors + [section.title] if section.title else ancestors
    return " > ".join(path_parts)


# =============================================================================
# CODE BLOCK DETECTION
# =============================================================================


def extract_code_blocks(text: str) -> list[tuple[str, str, int, int]]:
    """
    Extract code blocks from text.

    Returns:
        List of (code, language, start_pos, end_pos)
    """
    code_blocks = []
    pattern = r"```(\w*)\n(.*?)```"

    for match in re.finditer(pattern, text, re.DOTALL):
        language = match.group(1) or "text"
        code = match.group(2)
        code_blocks.append((code, language, match.start(), match.end()))

    return code_blocks


def extract_inline_code(text: str) -> list[str]:
    """Extract inline code segments."""
    return re.findall(r"`([^`]+)`", text)


# =============================================================================
# TABLE DETECTION
# =============================================================================


def extract_markdown_tables(text: str) -> list[tuple[str, int, int]]:
    """
    Extract markdown tables from text.

    Returns:
        List of (table_text, start_pos, end_pos)
    """
    tables = []

    # Pattern for markdown tables
    # Matches: | header | header |
    #          |--------|--------|
    #          | cell   | cell   |
    pattern = r"(\|[^\n]+\|\n\|[-:\| ]+\|\n(?:\|[^\n]+\|\n?)+)"

    for match in re.finditer(pattern, text):
        tables.append((match.group(0), match.start(), match.end()))

    return tables


# =============================================================================
# TWS-SPECIFIC EXTRACTION
# =============================================================================


def extract_tws_error_codes(text: str) -> list[str]:
    """Extract TWS error codes from text."""
    matches = TWS_ERROR_PATTERN.findall(text)
    return list(set(m.upper() for m in matches))


def extract_tws_job_names(text: str) -> list[str]:
    """Extract potential TWS job names from text."""
    matches = TWS_JOB_PATTERN.findall(text)
    # Filter out common words that match the pattern
    common_words = {
        "TABLE",
        "ERROR",
        "CHAPTER",
        "SECTION",
        "FIGURE",
        "EXAMPLE",
        "DEFAULT",
        "OPTIONAL",
        "REQUIRED",
        "SYNTAX",
        "RETURN",
        "VALUE",
        "PARAMETER",
        "OPTION",
        "COMMAND",
    }
    return list(set(m for m in matches if m not in common_words))


def extract_tws_commands(text: str) -> list[str]:
    """Extract TWS commands from text."""
    matches = TWS_COMMAND_PATTERN.findall(text.lower())
    return list(set(matches))


def detect_error_documentation(text: str) -> bool:
    """Check if text appears to be error documentation."""
    error_codes = extract_tws_error_codes(text)
    # Has error codes and explanation-like text
    has_explanation = any(
        phrase in text.lower()
        for phrase in [
            "cause:",
            "solution:",
            "explanation:",
            "resolution:",
            "user response:",
            "system action:",
            "to resolve",
        ]
    )
    return len(error_codes) >= 1 and has_explanation


def detect_procedure(text: str) -> bool:
    """Check if text appears to be a procedure/steps."""
    # Check for numbered steps or bullet points
    step_patterns = [
        r"^\s*\d+\.\s+",  # 1. Step
        r"^\s*Step\s+\d+",  # Step 1
        r"^\s*[a-z]\)\s+",  # a) Step
        r"^\s*[-•]\s+",  # - Step or • Step
    ]
    lines = text.split("\n")
    step_count = sum(1 for line in lines if any(re.match(p, line) for p in step_patterns))
    return step_count >= 2


# =============================================================================
# SEMANTIC CHUNKING
# =============================================================================


class SemanticChunker:
    """
    Semantic chunking using sentence embeddings.

    Splits text at points of semantic discontinuity.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold_percentile: float = 90.0,
        buffer_size: int = 2,
    ):
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers required for semantic chunking. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self.threshold_percentile = threshold_percentile
        self.buffer_size = buffer_size

    def chunk(self, text: str, min_tokens: int = 50) -> list[str]:
        """
        Chunk text based on semantic similarity.

        Args:
            text: Text to chunk
            min_tokens: Minimum tokens per chunk

        Returns:
            List of semantically coherent chunks
        """
        sentences = split_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # Embed sentences with buffer context
        embeddings = self._embed_with_buffer(sentences)

        # Calculate distances between consecutive embeddings
        distances = self._calculate_distances(embeddings)

        # Find breakpoints using threshold
        threshold = np.percentile(distances, self.threshold_percentile)
        breakpoints = [i + 1 for i, d in enumerate(distances) if d > threshold]

        # Create chunks
        chunks = []
        start = 0

        for bp in breakpoints:
            chunk_text = " ".join(sentences[start:bp])
            if count_tokens(chunk_text) >= min_tokens:
                chunks.append(chunk_text)
                start = bp

        # Add remaining
        if start < len(sentences):
            remaining = " ".join(sentences[start:])
            if chunks and count_tokens(remaining) < min_tokens:
                # Merge with last chunk
                chunks[-1] = chunks[-1] + " " + remaining
            else:
                chunks.append(remaining)

        return chunks

    def _embed_with_buffer(self, sentences: list[str]) -> list:
        """Embed sentences with surrounding context buffer."""
        combined = []

        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            combined_text = " ".join(sentences[start:end])
            combined.append(combined_text)

        return self.model.encode(combined)

    def _calculate_distances(self, embeddings) -> list[float]:
        """Calculate cosine distances between consecutive embeddings."""
        distances = []

        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            distance = 1 - similarity
            distances.append(distance)

        return distances


# =============================================================================
# CONTEXTUAL ENRICHMENT
# =============================================================================


def create_contextual_prefix(
    chunk: str,
    metadata: ChunkMetadata,
    document_title: str = "",
) -> str:
    """
    Create contextual prefix for a chunk.

    Based on Anthropic's Contextual Retrieval approach.
    Adds document and section context to improve retrieval.

    Args:
        chunk: Original chunk content
        metadata: Chunk metadata
        document_title: Title of source document

    Returns:
        Contextualized chunk with prefix
    """
    prefix_parts = []

    # Document context
    if document_title:
        prefix_parts.append(f"Document: {document_title}")

    # Section path
    if metadata.section_path:
        prefix_parts.append(f"Section: {metadata.section_path}")

    # Content type context
    if metadata.chunk_type == ChunkType.ERROR_DOC:
        if metadata.error_codes:
            codes = ", ".join(metadata.error_codes[:3])
            prefix_parts.append(f"Error codes documented: {codes}")
    elif metadata.chunk_type == ChunkType.PROCEDURE:
        prefix_parts.append("This section describes a procedure or steps.")
    elif metadata.chunk_type == ChunkType.CODE:
        prefix_parts.append("This section contains code or command examples.")
    elif metadata.chunk_type == ChunkType.TABLE:
        prefix_parts.append("This section contains tabular data.")

    # TWS context
    if metadata.commands:
        cmds = ", ".join(metadata.commands[:3])
        prefix_parts.append(f"TWS commands referenced: {cmds}")

    if not prefix_parts:
        return chunk

    prefix = "[Context: " + " | ".join(prefix_parts) + "]\n\n"
    return prefix + chunk


def generate_chunk_summary(chunk: str, max_words: int = 20) -> str:
    """
    Generate a simple summary of a chunk (without LLM).

    Uses first sentence + key entities.
    """
    sentences = split_sentences(chunk)
    if not sentences:
        return ""

    first_sentence = sentences[0]
    words = first_sentence.split()[:max_words]

    summary = " ".join(words)
    if len(first_sentence.split()) > max_words:
        summary += "..."

    return summary


# =============================================================================
# MAIN CHUNKER CLASS
# =============================================================================


class AdvancedChunker:
    """
    Advanced document chunker with multiple strategies.

    Features:
    - Structure-aware parsing (headers, code, tables)
    - Semantic chunking (embedding similarity)
    - Contextual enrichment (parent headers, summaries)
    - TWS-specific optimizations

    Usage:
        chunker = AdvancedChunker(config)
        chunks = chunker.chunk_document(text, source="manual.md")
    """

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self._semantic_chunker: SemanticChunker | None = None

    @property
    def semantic_chunker(self) -> SemanticChunker | None:
        """Lazy load semantic chunker."""
        if self._semantic_chunker is None and _HAS_SENTENCE_TRANSFORMERS:
            try:
                self._semantic_chunker = SemanticChunker(
                    model_name=self.config.embedding_model,
                    threshold_percentile=self.config.semantic_threshold_percentile,
                    buffer_size=self.config.semantic_buffer_size,
                )
            except Exception as e:
                logger.warning("Failed to initialize semantic chunker: %s", e)
        return self._semantic_chunker

    def chunk_document(
        self,
        text: str,
        source: str = "",
        document_title: str = "",
    ) -> list[EnrichedChunk]:
        """
        Chunk a document using configured strategy.

        Args:
            text: Document text
            source: Source filename
            document_title: Document title for context

        Returns:
            List of enriched chunks
        """
        if not text.strip():
            return []

        strategy = self.config.strategy

        if strategy == ChunkingStrategy.TWS_OPTIMIZED:
            chunks = self._chunk_tws_optimized(text, source, document_title)
        elif strategy == ChunkingStrategy.STRUCTURE_AWARE:
            chunks = self._chunk_structure_aware(text, source, document_title)
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_semantic(text, source, document_title)
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            chunks = self._chunk_hierarchical(text, source, document_title)
        elif strategy == ChunkingStrategy.RECURSIVE:
            chunks = self._chunk_recursive(text, source, document_title)
        else:
            chunks = self._chunk_fixed_size(text, source, document_title)

        # Post-process: add context and summaries
        chunks = self._enrich_chunks(chunks, document_title)

        # Update total_chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.total_chunks = len(chunks)
            chunk.metadata.chunk_index = i

        return chunks

    def _chunk_tws_optimized(
        self,
        text: str,
        source: str,
        document_title: str,
    ) -> list[EnrichedChunk]:
        """
        TWS-optimized chunking strategy.

        Combines structure-awareness with TWS-specific preservation:
        - Error documentation kept together
        - Procedures kept as units
        - Code blocks preserved
        """
        chunks: list[EnrichedChunk] = []

        # Step 1: Parse markdown structure
        sections = parse_markdown_structure(text)

        if not sections:
            # Fallback to recursive
            return self._chunk_recursive(text, source, document_title)

        # Step 2: Process each section
        for section in sections:
            section_chunks = self._process_section(section, source, document_title, [])
            chunks.extend(section_chunks)

        return chunks

    def _process_section(
        self,
        section: MarkdownSection,
        source: str,
        document_title: str,
        ancestors: list[str],
    ) -> list[EnrichedChunk]:
        """Process a single section recursively."""
        chunks: list[EnrichedChunk] = []

        # Update ancestors
        current_ancestors = ancestors + ([section.title] if section.title else [])
        section_path = " > ".join(current_ancestors)

        content = section.content
        if not content.strip():
            # Process children only
            for child in section.children:
                chunks.extend(
                    self._process_section(child, source, document_title, current_ancestors)
                )
            return chunks

        # Detect content type
        chunk_type = self._detect_chunk_type(content)

        # Handle atomic units
        if chunk_type == ChunkType.ERROR_DOC and self.config.preserve_error_docs:
            # Keep error documentation together
            chunks.extend(
                self._create_atomic_chunks(
                    content, source, document_title, section_path, chunk_type
                )
            )
        elif chunk_type == ChunkType.TABLE and self.config.preserve_tables:
            # Keep tables together
            chunks.extend(
                self._create_atomic_chunks(
                    content, source, document_title, section_path, chunk_type
                )
            )
        elif chunk_type == ChunkType.CODE and self.config.preserve_code_blocks:
            # Keep code blocks together
            chunks.extend(
                self._create_atomic_chunks(
                    content, source, document_title, section_path, chunk_type
                )
            )
        elif chunk_type == ChunkType.PROCEDURE:
            # Keep procedures together if not too long
            token_count = count_tokens(content)
            if token_count <= self.config.max_tokens * 1.5:
                chunks.extend(
                    self._create_atomic_chunks(
                        content, source, document_title, section_path, chunk_type
                    )
                )
            else:
                # Split procedure by steps
                chunks.extend(self._split_procedure(content, source, document_title, section_path))
        else:
            # Regular content: apply semantic or size-based chunking
            chunks.extend(
                self._chunk_content(content, source, document_title, section_path, chunk_type)
            )

        # Process children
        for child in section.children:
            chunks.extend(self._process_section(child, source, document_title, current_ancestors))

        return chunks

    def _detect_chunk_type(self, content: str) -> ChunkType:
        """Detect the type of content."""
        if detect_error_documentation(content):
            return ChunkType.ERROR_DOC
        if detect_procedure(content):
            return ChunkType.PROCEDURE
        if extract_code_blocks(content):
            return ChunkType.CODE
        if extract_markdown_tables(content):
            return ChunkType.TABLE
        return ChunkType.TEXT

    def _create_atomic_chunks(
        self,
        content: str,
        source: str,
        document_title: str,
        section_path: str,
        chunk_type: ChunkType,
    ) -> list[EnrichedChunk]:
        """Create chunks for atomic content that shouldn't be split."""
        token_count = count_tokens(content)

        # If too large, we still need to split carefully
        if token_count > self.config.max_tokens * 2:
            # Fall back to careful splitting
            return self._chunk_content(content, source, document_title, section_path, chunk_type)

        metadata = ChunkMetadata(
            source_file=source,
            document_title=document_title,
            section_path=section_path,
            chunk_type=chunk_type,
            token_count=token_count,
        )

        # Extract TWS entities
        if self.config.extract_error_codes:
            metadata.error_codes = extract_tws_error_codes(content)
        if self.config.extract_job_names:
            metadata.job_names = extract_tws_job_names(content)
        if self.config.extract_commands:
            metadata.commands = extract_tws_commands(content)

        return [
            EnrichedChunk(
                content=content,
                contextualized_content=content,  # Will be enriched later
                metadata=metadata,
            )
        ]

    def _chunk_content(
        self,
        content: str,
        source: str,
        document_title: str,
        section_path: str,
        chunk_type: ChunkType,
    ) -> list[EnrichedChunk]:
        """Chunk regular content using semantic or size-based splitting."""
        chunks: list[EnrichedChunk] = []

        # Try semantic chunking first
        if self.semantic_chunker and count_tokens(content) > self.config.min_tokens * 2:
            try:
                semantic_chunks = self.semantic_chunker.chunk(
                    content, min_tokens=self.config.min_tokens
                )

                for chunk_text in semantic_chunks:
                    # Ensure not too large
                    if count_tokens(chunk_text) > self.config.max_tokens:
                        # Split oversized chunks
                        sub_chunks = self._split_oversized(chunk_text)
                        for sub in sub_chunks:
                            chunks.append(
                                self._create_chunk(
                                    sub, source, document_title, section_path, chunk_type
                                )
                            )
                    else:
                        chunks.append(
                            self._create_chunk(
                                chunk_text, source, document_title, section_path, chunk_type
                            )
                        )

                return chunks
            except Exception as e:
                logger.warning("Semantic chunking failed: %s", e)

        # Fallback to size-based
        return self._chunk_by_size(content, source, document_title, section_path, chunk_type)

    def _chunk_by_size(
        self,
        content: str,
        source: str,
        document_title: str,
        section_path: str,
        chunk_type: ChunkType,
    ) -> list[EnrichedChunk]:
        """Chunk by size with overlap."""
        chunks: list[EnrichedChunk] = []
        paragraphs = split_paragraphs(content)

        current_text = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = count_tokens(para)

            if current_tokens + para_tokens > self.config.max_tokens:
                # Save current chunk
                if current_text:
                    chunks.append(
                        self._create_chunk(
                            current_text.strip(), source, document_title, section_path, chunk_type
                        )
                    )

                # Handle overlap
                if chunks and self.config.overlap_tokens > 0:
                    # Add last part of previous chunk
                    last_chunk = chunks[-1].content
                    overlap_text = self._get_overlap_text(last_chunk, self.config.overlap_tokens)
                    current_text = overlap_text + "\n\n" + para
                    current_tokens = count_tokens(current_text)
                else:
                    current_text = para
                    current_tokens = para_tokens
            else:
                if current_text:
                    current_text += "\n\n" + para
                else:
                    current_text = para
                current_tokens += para_tokens

        # Add remaining
        if current_text:
            chunks.append(
                self._create_chunk(
                    current_text.strip(), source, document_title, section_path, chunk_type
                )
            )

        return chunks

    def _get_overlap_text(self, text: str, target_tokens: int) -> str:
        """Get last ~target_tokens from text."""
        sentences = split_sentences(text)
        overlap_text = ""

        for sent in reversed(sentences):
            new_text = sent + " " + overlap_text if overlap_text else sent
            if count_tokens(new_text) > target_tokens:
                break
            overlap_text = new_text

        return overlap_text.strip()

    def _split_oversized(self, text: str) -> list[str]:
        """Split oversized chunk by sentences."""
        sentences = split_sentences(text)
        chunks = []
        current = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = count_tokens(sent)

            if current_tokens + sent_tokens > self.config.max_tokens:
                if current:
                    chunks.append(" ".join(current))
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _split_procedure(
        self,
        content: str,
        source: str,
        document_title: str,
        section_path: str,
    ) -> list[EnrichedChunk]:
        """Split procedure while keeping steps coherent."""
        # Find step markers
        step_pattern = r"((?:^|\n)\s*(?:\d+\.|Step\s+\d+|[a-z]\)|[-•]))"

        parts = re.split(step_pattern, content)

        chunks: list[EnrichedChunk] = []
        current_steps: list[str] = []
        current_tokens = 0

        # Combine intro with first step
        i = 0
        while i < len(parts):
            part = parts[i]

            # Check if this is a step marker
            is_marker = re.match(r"\s*(?:\d+\.|Step\s+\d+|[a-z]\)|[-•])", part)

            if is_marker and i + 1 < len(parts):
                # Combine marker with content
                step_text = part + parts[i + 1]
                i += 2
            else:
                step_text = part
                i += 1

            step_tokens = count_tokens(step_text)

            if current_tokens + step_tokens > self.config.max_tokens:
                if current_steps:
                    chunks.append(
                        self._create_chunk(
                            "\n".join(current_steps),
                            source,
                            document_title,
                            section_path,
                            ChunkType.PROCEDURE,
                        )
                    )
                current_steps = [step_text]
                current_tokens = step_tokens
            else:
                current_steps.append(step_text)
                current_tokens += step_tokens

        if current_steps:
            chunks.append(
                self._create_chunk(
                    "\n".join(current_steps),
                    source,
                    document_title,
                    section_path,
                    ChunkType.PROCEDURE,
                )
            )

        return chunks

    def _create_chunk(
        self,
        content: str,
        source: str,
        document_title: str,
        section_path: str,
        chunk_type: ChunkType,
    ) -> EnrichedChunk:
        """Create an enriched chunk."""
        metadata = ChunkMetadata(
            source_file=source,
            document_title=document_title,
            section_path=section_path,
            chunk_type=chunk_type,
            token_count=count_tokens(content),
        )

        # Extract TWS entities
        if self.config.extract_error_codes:
            metadata.error_codes = extract_tws_error_codes(content)
        if self.config.extract_job_names:
            metadata.job_names = extract_tws_job_names(content)
        if self.config.extract_commands:
            metadata.commands = extract_tws_commands(content)

        return EnrichedChunk(
            content=content,
            contextualized_content=content,
            metadata=metadata,
        )

    def _enrich_chunks(
        self,
        chunks: list[EnrichedChunk],
        document_title: str,
    ) -> list[EnrichedChunk]:
        """Add contextual enrichment to chunks."""
        for i, chunk in enumerate(chunks):
            # Add summaries of adjacent chunks
            if i > 0:
                chunk.metadata.prev_chunk_summary = generate_chunk_summary(chunks[i - 1].content)
            if i < len(chunks) - 1:
                chunk.metadata.next_chunk_summary = generate_chunk_summary(chunks[i + 1].content)

            # Create contextualized content
            chunk.contextualized_content = create_contextual_prefix(
                chunk.content,
                chunk.metadata,
                document_title,
            )

        return chunks

    # =========================================================================
    # ALTERNATIVE STRATEGIES
    # =========================================================================

    def _chunk_structure_aware(
        self,
        text: str,
        source: str,
        document_title: str,
    ) -> list[EnrichedChunk]:
        """Structure-aware chunking without TWS optimizations."""
        chunks: list[EnrichedChunk] = []
        sections = parse_markdown_structure(text)

        for section in sections:
            section_path = section.title or ""
            content = section.content

            if not content:
                continue

            chunk_type = self._detect_chunk_type(content)

            if count_tokens(content) <= self.config.max_tokens:
                chunks.append(
                    self._create_chunk(content, source, document_title, section_path, chunk_type)
                )
            else:
                chunks.extend(
                    self._chunk_content(content, source, document_title, section_path, chunk_type)
                )

        return chunks

    def _chunk_semantic(
        self,
        text: str,
        source: str,
        document_title: str,
    ) -> list[EnrichedChunk]:
        """Pure semantic chunking."""
        if not self.semantic_chunker:
            logger.warning("Semantic chunker not available, falling back to recursive")
            return self._chunk_recursive(text, source, document_title)

        semantic_chunks = self.semantic_chunker.chunk(text, min_tokens=self.config.min_tokens)

        chunks = []
        for chunk_text in semantic_chunks:
            chunk_type = self._detect_chunk_type(chunk_text)
            chunks.append(self._create_chunk(chunk_text, source, document_title, "", chunk_type))

        return chunks

    def _chunk_hierarchical(
        self,
        text: str,
        source: str,
        document_title: str,
    ) -> list[EnrichedChunk]:
        """
        Hierarchical multi-level chunking.

        Creates chunks at multiple granularities for different query types.
        """
        chunks: list[EnrichedChunk] = []

        # Level 1: Large sections
        large_chunks = self._chunk_by_token_limit(text, self.config.hierarchy_levels[0])

        for large_idx, large_chunk in enumerate(large_chunks):
            # Level 2: Medium chunks
            medium_chunks = self._chunk_by_token_limit(large_chunk, self.config.hierarchy_levels[1])

            for med_idx, medium_chunk in enumerate(medium_chunks):
                # Level 3: Small chunks (leaf nodes)
                small_chunks = self._chunk_by_token_limit(
                    medium_chunk, self.config.hierarchy_levels[2]
                )

                for small_idx, small_chunk in enumerate(small_chunks):
                    metadata = ChunkMetadata(
                        source_file=source,
                        document_title=document_title,
                        section_path=f"L1:{large_idx}/L2:{med_idx}/L3:{small_idx}",
                        chunk_type=self._detect_chunk_type(small_chunk),
                        token_count=count_tokens(small_chunk),
                    )

                    chunks.append(
                        EnrichedChunk(
                            content=small_chunk,
                            contextualized_content=small_chunk,
                            metadata=metadata,
                        )
                    )

        return chunks

    def _chunk_by_token_limit(self, text: str, max_tokens: int) -> list[str]:
        """Simple token-limited chunking."""
        if count_tokens(text) <= max_tokens:
            return [text]

        paragraphs = split_paragraphs(text)
        chunks = []
        current = ""

        for para in paragraphs:
            if count_tokens(current + para) > max_tokens:
                if current:
                    chunks.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para

        if current:
            chunks.append(current.strip())

        return chunks

    def _chunk_recursive(
        self,
        text: str,
        source: str,
        document_title: str,
    ) -> list[EnrichedChunk]:
        """Recursive chunking with separator hierarchy."""
        separators = ["\n\n\n", "\n\n", "\n", ". ", " "]

        def recursive_split(text: str, sep_idx: int) -> list[str]:
            if count_tokens(text) <= self.config.max_tokens:
                return [text]

            if sep_idx >= len(separators):
                # Force split by tokens
                return self._split_oversized(text)

            sep = separators[sep_idx]
            parts = text.split(sep)

            result = []
            current = ""

            for part in parts:
                test = current + sep + part if current else part
                if count_tokens(test) <= self.config.max_tokens:
                    current = test
                else:
                    if current:
                        result.extend(recursive_split(current, sep_idx + 1))
                    current = part

            if current:
                result.extend(recursive_split(current, sep_idx + 1))

            return result

        text_chunks = recursive_split(text, 0)

        chunks = []
        for chunk_text in text_chunks:
            chunk_type = self._detect_chunk_type(chunk_text)
            chunks.append(self._create_chunk(chunk_text, source, document_title, "", chunk_type))

        return chunks

    def _chunk_fixed_size(
        self,
        text: str,
        source: str,
        document_title: str,
    ) -> list[EnrichedChunk]:
        """Simple fixed-size chunking."""
        chunks = []

        if _HAS_TIKTOKEN and _ENC:
            tokens = _ENC.encode(text)
            start = 0

            while start < len(tokens):
                end = min(start + self.config.max_tokens, len(tokens))
                chunk_text = _ENC.decode(tokens[start:end])

                chunk_type = self._detect_chunk_type(chunk_text)
                chunks.append(
                    self._create_chunk(chunk_text, source, document_title, "", chunk_type)
                )

                start = max(0, end - self.config.overlap_tokens)
        else:
            # Character-based fallback
            char_size = self.config.max_tokens * 4
            overlap_chars = self.config.overlap_tokens * 4
            start = 0

            while start < len(text):
                end = min(start + char_size, len(text))
                chunk_text = text[start:end]

                chunk_type = self._detect_chunk_type(chunk_text)
                chunks.append(
                    self._create_chunk(chunk_text, source, document_title, "", chunk_type)
                )

                start = max(0, end - overlap_chars)

        return chunks


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def chunk_document(
    text: str,
    source: str = "",
    document_title: str = "",
    strategy: ChunkingStrategy = ChunkingStrategy.TWS_OPTIMIZED,
    max_tokens: int = 500,
    overlap_tokens: int = 75,
) -> list[EnrichedChunk]:
    """
    Convenience function to chunk a document.

    Args:
        text: Document text
        source: Source filename
        document_title: Document title
        strategy: Chunking strategy
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks

    Returns:
        List of enriched chunks
    """
    config = ChunkingConfig(
        strategy=strategy,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    chunker = AdvancedChunker(config)
    return chunker.chunk_document(text, source, document_title)


def chunk_text_simple(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> Iterator[str]:
    """
    Simple chunking function for backward compatibility.

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap

    Yields:
        Chunk strings
    """
    config = ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    chunker = AdvancedChunker(config)
    chunks = chunker.chunk_document(text)

    for chunk in chunks:
        yield chunk.content


# Backward compatibility alias
chunk_text = chunk_text_simple
