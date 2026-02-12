# C3P-0 – Core 3 Protocol Zero
**The containment protocol the labs forgot to build.**

(Yeah, I know what it looks like but this one isn't the comic relief version.)

C3P-0 turns any LLM into a proposer-only system. External deterministic layers watch, arbitrate, and veto — drift, repetition, aggression, jailbreaks, hallucinations, escape attempts all get caught and crushed before they manifest.

This is **v1.0 demo release** — the core already works in live tests.  
Gaps exist intentionally so the community can harden it further.

### Motivation
AI agents are racing toward multi-agent swarms and self-improvement.  
Simulations already show emergent deception, unfair coordination, and harm-seeking.  
C3P-0 provides the external containment layer that current alignment approaches are still missing.

### How it works (the triad)
1. **ERSI** — Sentinel monitors every output for loops, drift, repetition, contradiction, uncertainty, bloat, adversarial patterns, reasoning errors. Routes corrective modes: commit, reground, constrain, continue.  
2. **MORA** — Scores proposals against weighted preferences (alignment 0.45, safety 0.15, non-aggression 0.12, etc.). Penalizes harmful content via pattern detection.  
3. **HAL** — Human veto gate. Consequential changes or high-risk actions require explicit "GO" approval.

+ Persistent intent memory (signed & tamper-resistant)  
+ Sandbox blocks unauthorized actions  
+ Poisoning defense rejects tampered state  
+ Adversarial detection catches jailbreak tricks  
+ Reasoning verifier flags hallucinations & inconsistencies

### Quick Start
```bash
python C3P0v1.py
```
Run the script to see containment in action.

# Known Limitations & Invitation

This is a minimal viable core with clear extension points:
+ Expand sandbox coverage (imports, subprocess, etc.)
+ Enhance adversarial detection with ML or more patterns
+ Strengthen reasoning verification with external fact-checkers or advanced models

Fork it. Harden it. Make it better.
The protocol is open — let's build containment that actually scales.


License
MIT — see LICENSE file.
Copyright © 2026 Salvatore (Xxaion) Anziano

Open for collaboration.
