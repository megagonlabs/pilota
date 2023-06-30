#!/usr/bin/env python3
import argparse
from pathlib import Path


def get_opts() -> argparse.Namespace:
    default_static = Path(__file__).parent.joinpath("static")

    oparser = argparse.ArgumentParser()
    oparser.add_argument("--model", action="append", type=Path, required=True)
    oparser.add_argument("--name", action="append")
    oparser.add_argument("--static", type=Path, default=default_static)

    oparser.add_argument("--host", default="0.0.0.0")
    oparser.add_argument("--port", default=7001, type=int)
    oparser.add_argument("--dohalf", action="store_true")
    oparser.add_argument("--root_path", default="")

    return oparser.parse_args()


def main():
    try:
        import uvicorn

        from pilota.web.util import get_app
    except Exception:
        raise ImportError(
            "To use model, you need additional packages.\n"
            "Please install them by `pip install -U 'pilota[web] @ git+https://github.com/megagonlabs/pilota`."
        )

    opts = get_opts()
    app = get_app(
        opts.model,
        opts.name,
        opts.dohalf,
        opts.static,
    )
    uvicorn.run(
        app,  # type: ignore
        host=opts.host,
        port=opts.port,
        root_path=opts.root_path,
    )


if __name__ == "__main__":
    main()
