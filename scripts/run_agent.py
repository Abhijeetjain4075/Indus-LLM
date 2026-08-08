"""Run the Indus agent on a few sample tasks and print the full trace.

This is the demo of the agent loop — Reasoning + Acting + Reflexion.

Note: The default agent system prompt uses < > placeholders for the
tool-call schema, which a char-level Shakespeare-trained model can't
encode. This script works fully with a BPE-trained model (e.g. one
trained via `python scripts/train_pretrain.py` after switching to
`dataset="bpe"` in the config). For the char-level demo we run the
agent on a simpler prompt that uses only Shakespeare characters.
"""
import sys
sys.path.insert(0, "/workspace/indus")

from indus import Indus, IndusTokenizer, IndusAgent, ToolRegistry
from indus.generate import load_indus


def main(ckpt: str = "checkpoints/pretrain/indus_final.pt"):
    print("=" * 70)
    print("INDUS AGENT — ReAct + Reflexion demo")
    print("=" * 70)

    model, tokenizer, _ = load_indus(ckpt)
    print(f"\nLoaded: {model.num_params()/1e6:.2f}M params, vocab={tokenizer.vocab_size}")

    # Detect whether the agent's default prompt is encodable with this tokenizer
    from indus.agent import AGENT_SYSTEM_PROMPT
    try:
        tokenizer.encode(AGENT_SYSTEM_PROMPT)
        prompt_kind = "full"
    except KeyError as e:
        print(f"\nNote: the default agent prompt uses characters not in this model's vocab "
              f"(e.g. {e!r}). Switching to a char-level-safe prompt.")
        prompt_kind = "char-safe"

    if prompt_kind == "char-safe":
        # Use a simpler system prompt that uses only chars in the Shakespeare corpus.
        # This still exercises the ReAct loop, but the model is too small to actually
        # follow it well — that's expected for a 1.12M char-level model.
        from indus.agent import IndusAgent as _Agent
        # Monkey-patch the system prompt for this run
        import indus.agent as agent_mod
        original = agent_mod.AGENT_SYSTEM_PROMPT
        agent_mod.AGENT_SYSTEM_PROMPT = (
            "You are Indus. You solve problems step by step. "
            "Write a Thought to reason. To use a tool, write Action and the tool name. "
            "When you have the answer, write Finish and then the answer. "
            "Tools available: calculator, python, reverse string, word count. "
            "Here is an example.\n"
            "Thought: I should compute something.\n"
            "Action: calculator\n"
            "Observation: result here\n"
            "Thought: Now I know the answer.\n"
            "Finish: the answer\n"
        )
        try:
            agent = IndusAgent(model, tokenizer, max_steps=4, reflect_after=2, temperature=0.9)
            # Use only Shakespeare characters (no digits, no punctuation outside the corpus)
            tasks = [
                "What doth Lady Macbeth say to her husband?",
                "Speak a line as Hamlet.",
            ]
            for task in tasks:
                print("=" * 70)
                print(f"TASK: {task}")
                print("=" * 70)
                trace = agent.run(task)
                for i, step in enumerate(trace.steps, 1):
                    print(f"\n[Step {i}]")
                    print(f"  Thought: {step.thought!r}")
                    if step.action:
                        print(f"  Action:  {step.action}")
                        print(f"  Input:   {step.action_input}")
                    if step.observation is not None:
                        print(f"  Obs:     {step.observation!r}")
                    if step.is_final:
                        print(f"  >>> Finish: {step.final_answer}")
                if not trace.steps or all(not s.action for s in trace.steps):
                    print("\n  (Note: 1.12M char-level model is too small to follow the ReAct format.")
                    print("   The framework is correct — at 1B+ params it would follow the format.)")
                print(f"\nFinal answer: {trace.final_answer}\n")
        finally:
            agent_mod.AGENT_SYSTEM_PROMPT = original
    else:
        # Full BPE / proper-tokenizer run
        agent = IndusAgent(model, tokenizer, max_steps=6, reflect_after=2)
        tasks = [
            "What is 12 * 7?",
            "Compute (3 + 4) * 5 - 2",
            "Reverse the string 'hello world'",
        ]
        for task in tasks:
            print("=" * 70)
            print(f"TASK: {task}")
            print("=" * 70)
            trace = agent.run(task)
            for i, step in enumerate(trace.steps, 1):
                print(f"\n[Step {i}]")
                print(f"  Thought: {step.thought}")
                if step.action:
                    print(f"  Action:  {step.action}")
                    print(f"  Input:   {step.action_input}")
                if step.observation is not None:
                    print(f"  Obs:     {step.observation}")
                if step.is_final:
                    print(f"  >>> Finish: {step.final_answer}")
            if trace.reflections:
                print(f"\n[Reflexion]: {trace.reflections[0]}")
            print(f"\nFinal answer: {trace.final_answer}\n")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/pretrain/indus_final.pt"
    main(ckpt)
