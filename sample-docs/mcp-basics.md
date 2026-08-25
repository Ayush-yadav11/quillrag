# Understanding MCP Servers

The Model Context Protocol (MCP) is an open protocol that lets AI assistants
talk to external tools and data sources. An MCP server exposes three primitives:
tools (callable functions), resources (readable data), and prompts (templates).

## Stdio transport

Most desktop editors launch MCP servers as child processes and speak JSON-RPC
over stdin/stdout. Every message is one line of JSON. Logs must go to stderr,
never stdout, or the protocol stream gets corrupted.

## Tool design

A good tool has a crisp description, typed parameters with per-field docs, and
returns plain text the model can quote directly. Keep result payloads small:
return the top chunks, not entire documents.
