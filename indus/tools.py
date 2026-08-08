"""Built-in tools for the Indus agent.

Each tool has:
  - name:        short identifier
  - description: shown to the LLM in the system prompt
  - signature:   JSON schema for the arguments
  - run(args) -> str: the actual implementation

Indus can call any of these via the ReAct loop. The tool descriptions
are designed to be unambiguous so even a small model can pick correctly.
"""

from __future__ import annotations
import ast
import math
import operator
from typing import Any, Callable, Dict, List

from .model import Indus


class Tool:
    """A tool the agent can call."""
    def __init__(self, name: str, description: str, signature: Dict[str, Any],
                 fn: Callable[[Dict[str, Any]], str]):
        self.name = name
        self.description = description
        self.signature = signature
        self.fn = fn

    def call(self, args: Dict[str, Any]) -> str:
        try:
            return self.fn(args)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Calculator — safe AST-based arithmetic evaluator
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_SAFE_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
}


def _safe_eval(expr: str) -> Any:
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported op: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
            fn = _SAFE_FUNCS[node.func.id]
            if callable(fn):
                return fn(*[_eval_node(a) for a in node.args])
            return fn  # constant
        raise ValueError(f"unsupported call: {ast.dump(node.func)}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCS:
            v = _SAFE_FUNCS[node.id]
            return v() if callable(v) else v
        raise ValueError(f"unknown name: {node.id}")
    raise ValueError(f"unsupported node: {ast.dump(node)}")


def calculator_tool(args: Dict[str, Any]) -> str:
    """Evaluate a math expression safely."""
    expr = args.get("expression", "")
    if not expr:
        return "ERROR: empty expression"
    return str(_safe_eval(expr))


# ---------------------------------------------------------------------------
# Python sandbox — execute a short snippet and return stdout
# ---------------------------------------------------------------------------

def python_tool(args: Dict[str, Any]) -> str:
    """Run a small Python snippet and return captured stdout.

    Restricted: no imports of os, sys, subprocess, etc. This is a
    sandbox for evaluation, not a security boundary.
    """
    import io
    import contextlib
    code = args.get("code", "")
    if not code:
        return "ERROR: empty code"
    if any(banned in code for banned in ["import os", "import sys",
                                          "subprocess", "__import__",
                                          "open(", "exec(", "eval("]):
        return "ERROR: blocked import / call"
    buf = io.StringIO()
    g: Dict[str, Any] = {"__builtins__": __builtins__}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, g)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    return buf.getvalue() or "(no output)"


# ---------------------------------------------------------------------------
# String tools — simple text utilities
# ---------------------------------------------------------------------------

def reverse_tool(args: Dict[str, Any]) -> str:
    return args.get("text", "")[::-1]


def word_count_tool(args: Dict[str, Any]) -> str:
    text = args.get("text", "")
    return str(len(text.split()))


# ---------------------------------------------------------------------------
# Built-in registry
# ---------------------------------------------------------------------------

def default_tools() -> List[Tool]:
    return [
        Tool(
            name="calculator",
            description=("Evaluate a math expression and return the result. "
                         "Input: {\"expression\": \"<expr>\"}  e.g. 2 + 3 * 4"),
            signature={"expression": "string"},
            fn=calculator_tool,
        ),
        Tool(
            name="python",
            description=("Execute a short Python snippet and return its stdout. "
                         "Input: {\"code\": \"print(2+2)\"}"),
            signature={"code": "string"},
            fn=python_tool,
        ),
        Tool(
            name="reverse_string",
            description="Reverse a string. Input: {\"text\": \"hello\"}",
            signature={"text": "string"},
            fn=reverse_tool,
        ),
        Tool(
            name="word_count",
            description="Count words in a text. Input: {\"text\": \"<text>\"}",
            signature={"text": "string"},
            fn=word_count_tool,
        ),
    ]


class ToolRegistry:
    """A name -> Tool map the agent uses to look up and call tools."""
    def __init__(self, tools: List[Tool] = None):
        self.tools = {t.name: t for t in (tools or default_tools())}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def call(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self.tools:
            return f"ERROR: unknown tool '{name}'"
        return self.tools[name].call(args)

    def descriptions(self) -> str:
        """Render all tools as a system-prompt block."""
        out = ["Available tools:"]
        for t in self.tools.values():
            out.append(f"  - {t.name}: {t.description}")
        return "\n".join(out)

    def names(self) -> List[str]:
        return list(self.tools.keys())
