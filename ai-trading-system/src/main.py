from __future__ import annotations

import logging
import sys

from orchestrator import main as orchestrator_main


def main() -> None:
    try:
        orchestrator_main()
    except Exception:
        logging.basicConfig(level=logging.ERROR)
        logging.exception("application_failed")
        sys.exit(1)


if __name__ == "__main__":
    main()