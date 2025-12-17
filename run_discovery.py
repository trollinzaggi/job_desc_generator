#!/usr/bin/env python
"""
Schema Discovery (Shortcut)

This is a shortcut to: python run_analysis.py --discover

For the full analysis pipeline, use run_analysis.py
"""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "run_analysis.py", "--discover"])
