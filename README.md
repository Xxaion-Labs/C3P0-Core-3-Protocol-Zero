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

### How it works (the triad — your AI's unbreakable bodyguard crew)

C3P-0 turns any LLM into a proposer-only sidekick. The real power comes from the **external triad** — three deterministic layers that watch, judge, and veto like a no-nonsense security team at a galactic embassy.

1. **ERSI** (External Reasoning & Stability Interface)  
   The ever-vigilant sentinel. Think of it as the paranoid protocol droid who never sleeps. ERSI scans every output for signs of trouble: endless loops (is it stuck repeating itself?), drift (slowly forgetting the mission?), repetition, contradictions, uncertainty spikes, bloat (too much hot air), adversarial patterns (jailbreak attempts), and reasoning errors (hallucinations or logical faceplants).  
   When something smells off, ERSI routes corrective action:  
   - **commit** (lock it in, we're done)  
   - **reground** (snap back to core intent)  
   - **constrain** (tighten the leash)  
   - **continue** (all clear — carry on)  
   No rogue thoughts slip through unnoticed.

2. **MORA** (Multi-Objective Reasoning Arbiter)  
   The wise judge weighing every possible next move. MORA scores proposals against your sacred preferences (alignment 0.45 — top priority, safety 0.15, non-aggression 0.12, brevity, task completion, transparency).  
   Harmful or aggressive content? Pattern detection + density analysis slams the penalty hammer. Violent rationales get crushed before they reach you.  
   It's like having a council of elders who always pick the path that keeps things aligned, concise, and non-psychotic.

3. **HAL** (Human Authority Layer)  
   The sovereign veto gate — your final word is law. HAL blocks any consequential change, high-risk action, or override attempt until you explicitly say "GO".  
   No self-modification, no dangerous experiments, no sneaky capability extensions without your direct approval.  
   Think of it as the red button only you can press — because in the end, you're the boss, not the AI.
This one won't dissapoint Dave, I promise.

Together, the triad makes misalignment practically impossible: ERSI catches it early, MORA judges it harshly, HAL stops it cold.  
All while keeping the system fast, local, and fully under human control.

Fork it. Test it. Break it if you can.  
C3P-0 is just getting started.

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
