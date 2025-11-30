from __future__ import annotations

from sportsbd.cli import main

if __name__ == "__main__":
    import sys

    main(["sweep", *sys.argv[1:]])


