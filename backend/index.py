"""
Vercel entry point for the PrintChakra API.

`app.py` (the local runner) is excluded from the deployment because its name clashes with the
`app/` package. Locally, keep using `python app.py`.
"""

import os
import sys

# Make the `app` package importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core import create_app  # noqa: E402

app = create_app("production")
