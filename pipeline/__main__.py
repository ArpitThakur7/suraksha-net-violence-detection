"""CLI: python -m pipeline <subcommand> [args]. Modules are imported lazily."""
import importlib
import sys

SUBCOMMANDS = ("audit", "split", "fetch", "cache", "train", "evaluate", "grid", "bias", "bench", "probe", "figures", "report")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m pipeline <" + "|".join(SUBCOMMANDS) + "> [args]")
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd not in SUBCOMMANDS:
        print(f"unknown subcommand {cmd!r}; choose from {', '.join(SUBCOMMANDS)}", file=sys.stderr)
        return 2
    modname = f"pipeline.{cmd}"
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name == modname:
            print(f"subcommand {cmd!r} not implemented yet (missing {modname})", file=sys.stderr)
            return 3
        raise
    return mod.main(rest) or 0


if __name__ == "__main__":
    sys.exit(main())
