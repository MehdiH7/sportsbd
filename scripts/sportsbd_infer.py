from __future__ import annotations

from sportsbd.cli import main

if __name__ == "__main__":
    # Delegate to the main CLI with 'infer' as the subcommand.
    import sys

    main(["infer", *sys.argv[1:]])


