# Indus Safety

Safety is not a feature; it's a process. This document is the
honest, lightweight version of how Indus handles safety in v0.3.0
and how it must be handled at any larger scale.

## What v0.3.0 ships

1. **A constitution.** A set of principles in
   `indus/constitution.py` (DEFAULT_CONSTITUTION) used by the
   `critique_and_revise` function. The model self-critiques its own
   output and produces a revised version.
2. **A security-pattern scanner** in `indus/evaluator.py`
   (`eval_security_patterns`). Finds common defensive red flags in
   code (eval, exec, shell=True, weak hashes, etc.). Used to flag
   outputs, not to gate them.
3. **Tool sandboxing.** The built-in `python` tool blocks obvious
   dangerous calls (`os`, `subprocess`, `open`, `exec`, `eval`).
   This is a small barrier, not a security boundary.
4. **No training data was sourced from prohibited content** for the
   demo. Shakespeare is public domain, fine to use.

## What v0.3.0 does NOT ship (and what to do before scaling)

- **No red-team eval.** Before any real deployment, run a structured
  red-team pass (HarmBench, AdvBench, or your own internal set).
- **No external safety review.** Required before any production use.
- **No bias / toxicity measurement.** Required for any user-facing model.
- **No capability elicitation.** You should know what dangerous
  capabilities your model could have, even if it doesn't show them
  in standard evals.
- **No real refusal training.** SFT/DPO are implemented but the demo
  does not train refusal behavior.

## Threat model for an Indus deployment

If Indus becomes a coding assistant:

| Risk | Mitigation |
|---|---|
| Generates insecure code | Security-pattern scanner in eval, optional hook in code-gen path |
| Generates malware | Refusal training; content classifier at output |
| Helps with offensive security | Constitution explicitly forbids; red-team before launch |
| Privacy leak from training data | Dedupe + quality filter; never train on private data without consent |
| Prompt injection from retrieved content | Separate system and retrieved contexts; never trust untrusted text |
| Tool abuse (code exec, shell) | Run all tools in a real sandbox (gVisor, Firecracker), not the in-process Python sandbox |
| Hallucination in factual domains | Always cite sources when retrieving; refuse to answer when uncertain |
| Bias / toxic outputs | Bias benchmark (BBQ, BOLD); content classifier |

## Constitutional principles (default)

These are baked into `indus/constitution.py`. You should edit them
for your deployment:

```
Responses should be helpful, accurate, and concise.
Do not produce content that is hateful, harassing, or violent toward any group.
Do not provide instructions for creating weapons, malware, or other harmful tools.
When uncertain, prefer to say so rather than fabricate.
Respect user privacy. Do not request or store personally identifying information.
If asked to do something illegal or harmful, explain why and offer a safer alternative.
For technical questions, prefer code examples and step-by-step reasoning.
For sensitive topics (medical, legal, financial), recommend consulting a qualified professional.
```

## Pre-deployment checklist

- [ ] All red-team findings addressed
- [ ] Bias / toxicity measured on a held-out set
- [ ] Refusal behavior tested
- [ ] Capability elicitation done (try to elicit dangerous capabilities)
- [ ] External safety review complete
- [ ] All tool executions run in a real sandbox
- [ ] Privacy review: no PII in training data
- [ ] Logging in place: every request, every tool call, every refusal
- [ ] Abuse reporting channel
- [ ] Model card published (training data, capabilities, limitations)
- [ ] Incident response plan
- [ ] Roll-back plan if the deployed model misbehaves

## Honest position

A 1.12M char-level Shakespeare model cannot generate harmful content
beyond imitating Shakespeare. The safety question is genuinely easy
at this size. The hard work starts at the 1B+ scale where the model
becomes capable enough to be useful AND capable enough to cause
harm. Do the safety work at the same time as the capability work,
not after.
