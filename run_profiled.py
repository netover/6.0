#!/usr/bin/env python3.14
"""Simple profiler launcher for Resync"""
import sys
import uvicorn

if __name__ == "__main__":
    sys.exit(uvicorn.main(["resync.main:app", "--host", "0.0.0.0", "--port", "8000"]))
