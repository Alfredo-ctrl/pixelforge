"""pixelforge command line. Built for agents: JSON in, JSON out."""

import argparse
import json
import sys

from . import __version__, apply_ops, list_ops, load_image, save_image


def _fail(msg):
    print(json.dumps({"error": msg}), file=sys.stderr)
    raise SystemExit(1)


def _ok(payload):
    print(json.dumps(payload))


def cmd_edit(args):
    try:
        ops = json.loads(args.ops)
        img = load_image(args.input)
        out, applied = apply_ops(img, ops)
        fmt = save_image(out, args.output, quality=args.quality)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as e:
        _fail(str(e))
    _ok({"output": args.output, "format": fmt, "size": list(out.size), "ops_applied": applied})


def cmd_ops(args):
    _ok(list_ops())


def cmd_describe(args):
    try:
        img = load_image(args.input)
    except (ValueError, OSError) as e:
        _fail(str(e))
    _ok({"format": img.format, "mode": img.mode, "size": list(img.size)})


def cmd_mcp(args):
    from .mcp_server import run

    run()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pixelforge", description="Image editing for AI agents.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_edit = sub.add_parser("edit", help="Apply operations to an image and save the result.")
    p_edit.add_argument("-i", "--input", required=True, help="Input image path")
    p_edit.add_argument("-o", "--output", required=True, help="Output image path")
    p_edit.add_argument("--ops", required=True, help='JSON list of ops, e.g. \'[{"op": "exposure", "ev": 0.5}]\'')
    p_edit.add_argument("--quality", type=int, default=90, help="JPEG/WebP quality, 1-100")
    p_edit.set_defaults(func=cmd_edit)

    p_ops = sub.add_parser("ops", help="List every operation with its parameters.")
    p_ops.set_defaults(func=cmd_ops)

    p_desc = sub.add_parser("describe", help="Print image info as JSON.")
    p_desc.add_argument("-i", "--input", required=True, help="Input image path")
    p_desc.set_defaults(func=cmd_describe)

    p_mcp = sub.add_parser("mcp", help="Start the MCP server on stdio.")
    p_mcp.set_defaults(func=cmd_mcp)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
