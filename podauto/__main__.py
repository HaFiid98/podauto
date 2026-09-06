"""Entry point for the podauto package when run with `python -m podauto`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
