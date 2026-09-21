"""Entry point for `python -m doceval` and for the frozen binary.

PyInstaller builds against this module, so the packaged binary and the
installed console script run the same code path.
"""

from doceval.cli import main

if __name__ == "__main__":
    main()
