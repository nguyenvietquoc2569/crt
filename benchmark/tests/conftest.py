"""Shared pytest setup for the CRT journal tooling in benchmark/. Run on the Mac (no torch, no GPU):

    /Users/harry/duplex/.venv-stats/bin/python -m pytest benchmark/tests

The modules under test live one directory up and are imported by name, as the HPC job scripts run them.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)
