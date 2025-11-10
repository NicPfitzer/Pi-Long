from pilong import Pi_Long  # noqa: F401 - re-export for backwards compatibility
from pilong.cli import main, run_from_cli

__all__ = ["Pi_Long", "run_from_cli", "main"]


if __name__ == "__main__":
    main()

