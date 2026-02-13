# RAG Base Collection

This folder contains the high‑priority documents and API specification used by
Resync's retrieval‑augmented generation (RAG) layer. The files in this
collection are ingested into the vector database with a high priority so that
questions about the TWS API, AWS administration, troubleshooting and
configuration are answered from these sources before falling back to other
collections.

To ingest this collection, read `manifest.base.json` for a list of files and
associated tags, and `ingestion.base.json` for recommended chunking and
priority settings.
