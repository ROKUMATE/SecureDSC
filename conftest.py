"""Ensures the project root is importable so ``pytest`` (run as the entry-point
script, not just ``python -m pytest``) can import the ``securedsc`` package.
The mere presence of this file at the repo root puts the root on ``sys.path``.
"""
