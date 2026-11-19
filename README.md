# pixelforge

Image editing for AI agents. No GUI, no account, no vendor lock-in: a CLI, a Python API and an MCP server that any agent can drive — Muse, Antigravity, Codex, Claude Code, or a local model.

## Install

```sh
pip install pixelforge
```

Needs Python 3.10+. Background removal is optional:

```sh
pip install pixelforge[bg]
```

## Use it from the terminal

```sh
# what can it do?
pixelforge ops

# describe a file
pixelforge describe photo.jpg

# edit it: warmer, a bit brighter, gentle vignette
pixelforge edit -i photo.jpg -o photo-edit.jpg --ops '[
  {"op": "temperature", "value": 18},
  {"op": "exposure", "ev": 0.3},
  {"op": "vignette", "strength": 25}
]'
```

Output is JSON, so it is easy to parse:

```json
{"output": "photo-edit.jpg", "format": "JPEG", "size": [4000, 3000], "ops_applied": ["temperature", "exposure", "vignette"]}
```

## How an agent uses it

You translate the user's words into ops. `pixelforge ops` lists every operation with its parameters and valid ranges, so there is nothing to guess. Examples:

- "make it warmer and brighter" -> temperature +15, exposure +0.3
- "punchy, high contrast black and white" -> grayscale, contrast 1.3, clarity 40
- "soften the whole frame" -> blur 3

If the user asks for something that is not an op yet, add it — see CONTRIBUTING.md. That is the whole point: the tool grows with what people ask for.

## Use it from Python

```python
from pixelforge import load_image, save_image, apply_ops

img = load_image("photo.jpg")
out, applied = apply_ops(img, [{"op": "saturation", "factor": 1.2}])
save_image(out, "photo-edit.jpg")
```

## MCP server

Any MCP client can use pixelforge as tools:

```sh
pixelforge mcp
```

It speaks MCP over stdio and exposes `edit_image`, `describe_image` and `list_operations`. Example client config:

```json
{
  "mcpServers": {
    "pixelforge": {"command": "pixelforge", "args": ["mcp"]}
  }
}
```

## License

MIT
