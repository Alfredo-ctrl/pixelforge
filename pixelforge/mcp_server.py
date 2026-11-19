"""Minimal MCP server over stdio. No dependencies, plain JSON-RPC.

Tools: edit_image, describe_image, list_operations.
"""

import json
import sys

from . import __version__, apply_ops, list_ops, load_image, save_image

TOOLS = [
    {
        "name": "edit_image",
        "description": "Apply a list of image operations and save the result. Ops are the same ones list_operations returns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input image path"},
                "output": {"type": "string", "description": "Output image path"},
                "ops": {"type": "array", "items": {"type": "object"}, "description": "Ops to apply, e.g. [{\"op\": \"exposure\", \"ev\": 0.5}]"},
                "quality": {"type": "number", "description": "JPEG/WebP quality, 1-100"},
            },
            "required": ["input", "output", "ops"],
        },
    },
    {
        "name": "describe_image",
        "description": "Get format, mode and size of an image.",
        "inputSchema": {
            "type": "object",
            "properties": {"input": {"type": "string", "description": "Image path"}},
            "required": ["input"],
        },
    },
    {
        "name": "list_operations",
        "description": "List every operation with its parameters and valid ranges.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _text(payload):
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


def _call(name, args):
    if name == "list_operations":
        return _text(list_ops())
    if name == "describe_image":
        img = load_image(args["input"])
        return _text({"format": img.format, "mode": img.mode, "size": list(img.size)})
    if name == "edit_image":
        img = load_image(args["input"])
        out, applied = apply_ops(img, args["ops"])
        fmt = save_image(out, args["output"], quality=args.get("quality", 90))
        return _text({"output": args["output"], "format": fmt, "size": list(out.size), "ops_applied": applied})
    raise ValueError(f"unknown tool '{name}'")


def run():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method", "")
        req_id = req.get("id")
        if method in ("notifications/initialized", "notifications/cancelled"):
            continue
        if method == "initialize":
            result = {
                "protocolVersion": req.get("params", {}).get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "pixelforge", "version": __version__},
            }
            resp = {"result": result}
        elif method == "ping":
            resp = {"result": {}}
        elif method == "tools/list":
            resp = {"result": {"tools": TOOLS}}
        elif method == "tools/call":
            try:
                resp = {"result": _call(req["params"]["name"], req["params"].get("arguments", {}))}
            except Exception as e:
                resp = {"result": {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}], "isError": True}}
        else:
            resp = {"error": {"code": -32601, "message": f"method not found: {method}"}}
        if req_id is not None:
            print(json.dumps({"jsonrpc": "2.0", "id": req_id, **resp}), flush=True)
