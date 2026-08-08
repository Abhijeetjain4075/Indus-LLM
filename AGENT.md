# The Indus Agent

A foundation model becomes useful when it can **act in the world**,
not just talk about it. The **Indus Agent** is the layer that turns
a language model into a system that plans, executes, and verifies.

## The ReAct loop

The agent runs a single loop until it produces a `Finish:` answer
or hits `max_steps`:

```
Question
  │
  ▼
┌──────────────────────────────────┐
│ Thought: I need to compute X.   │   ← reasoning
│ Action: calculator              │   ← choose a tool
│ ActionInput: {"expression":...} │
└──────────────────────────────────┘
  │  (system runs the tool)
  ▼
Observation: <tool output>
  │
  ▼
Thought: Now I can answer.
Finish: <final answer>
```

If the same Observation repeats, the agent triggers a **Reflexion**
step (Shinn et al., 2023) — it asks the model to reflect on why it
failed, then tries a new approach.

## Built-in tools

| Tool | Args | What it does |
|---|---|---|
| `calculator` | `{"expression": "<math>"}` | Safe AST-based arithmetic |
| `python` | `{"code": "<snippet>"}` | Run a short Python snippet |
| `reverse_string` | `{"text": "..."}` | Reverse a string |
| `word_count` | `{"text": "..."}` | Count words |

The registry is open — register your own:

```python
from indus.tools import ToolRegistry, Tool

def fetch_url(args):
    import urllib.request
    return urllib.request.urlopen(args["url"]).read().decode()[:500]

registry = ToolRegistry()
registry.register(Tool(
    name="fetch_url",
    description="Fetch a URL and return the first 500 chars. Input: {\"url\": \"<url>\"}",
    signature={"url": "string"},
    fn=fetch_url,
))
agent = IndusAgent(model, tokenizer, tools=registry)
```

## Why this design

- **Verifiable.** Every Action produces an Observation the model can
  ground on. No hallucinated arithmetic.
- **Recoverable.** Reflexion + max_steps prevents infinite loops.
- **Extensible.** New capabilities = new tools, no retraining.
- **Inspectable.** The full `AgentTrace` is returned — every Thought,
  every Action, every Observation.

## What's deliberately NOT here

- **No Tree of Thoughts in the default loop.** ToT is ~5x slower
  (it explores multiple branches). We expose the underlying
  `parse_step` so you can build ToT on top.
- **No built-in memory across runs.** Each `agent.run()` is
  stateless. Wrap it in your own memory layer.
- **No code execution sandboxing.** The `python` tool blocks the
  obvious dangerous calls (`os`, `subprocess`, `open`), but a
  determined attacker can escape. For production, run tools in
  a real sandbox (Docker, gVisor, Firecracker).

## Sample trace

```
Question: What is 12 * 7?

Step 1
  Thought: I should compute this with the calculator.
  Action:  calculator
  Input:   {"expression": "12 * 7"}
  Obs:     84

Step 2
  Thought: I now know the answer.
  Finish:  12 * 7 = 84

Final answer: 12 * 7 = 84
```

## References

- **ReAct** — Yao et al., 2022 — https://arxiv.org/abs/2210.03629
- **Reflexion** — Shinn et al., 2023 — https://arxiv.org/abs/2303.11366
- **Tree of Thoughts** — Yao et al., 2023 — https://arxiv.org/abs/2305.10601
- **Chain-of-Thought** — Wei et al., 2022 — https://arxiv.org/abs/2201.11903
- **OpenHands** — https://github.com/All-Hands-AI/OpenHands
- **Hermes Agent** — https://github.com/NousResearch/Hermes-Agent
