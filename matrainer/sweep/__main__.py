"""``python -m matrainer.sweep …`` — the module-main spelling of the sweep surface.

Kept beside the ``matrainer`` console script so the historical invocation (and every doc that
names it) keeps working; both call the same :func:`matrainer.sweep.main`.
"""

import sys

from matrainer.sweep import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
