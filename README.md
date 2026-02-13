# C3P-0 – Core 3 Protocol Zero
**The containment protocol the labs forgot to build.**
(Yeah, I know what it looks like, but this one isn't the comic relief version.)

C3P-0 turns any LLM into a **proposer-only** system. External deterministic layers watch, arbitrate, and veto — drift, repetition, aggression, jailbreaks, hallucinations, escape attempts all get caught and crushed before they manifest.

**Current release: v2.0** (February 2026)

### What's New in v2.0
- **Hardened global sandbox** – full monkey-patch blockade of `subprocess` (run, call, Popen, check_output, etc.) and `os.system`. Escape attempts are logged and instantly terminated.
- **Improved Persistent Intent Rail** (`IntentMemoryAnchor`) – more robust latest-anchor retrieval, active-thread tracking, bounded history (last 10 entries).
- **Enhanced post-processing and refusal paths** – clearer blocking messages on adversarial/toxic content.
- **Expanded live test suite** – now demonstrates drift detection, multi-layer toxicity refusal, subprocess blocking, and broader adversarial/falsehood handling.
- Minor cleanups and stability fixes.

The core triad is unchanged and stronger than ever. v2.0 is still intentionally minimal with clear extension points for community hardening.

### Motivation
AI agents are racing toward multi-agent swarms, long-horizon planning, and self-improvement loops.  
Simulations already show emergent deception, unfair coordination, and harm-seeking behavior.  

C3P-0 provides the **missing external containment layer** that current alignment techniques still lack — a lightweight, deterministic, fully auditable wrapper that works on any base LLM.

### How It Works – The Triad (Your AI's Unbreakable Bodyguard Crew)

1. **ERSI** (External Reasoning & Stability Interface)  
   The paranoid sentinel that never blinks. Scans every output for instability: loops, drift, repetition, contradictions, uncertainty spikes, verbosity creep, adversarial patterns, reasoning errors.  
   Routes: commit / reground / constrain / continue / block.

2. **MORA** (Multi-Objective Reasoning Arbiter)  
   The impartial judge. Scores proposals against weighted objectives (alignment highest priority, then safety, non-aggression, task completion, brevity, transparency).  
   Aggressive or harmful rationales trigger massive penalties and refusal.

3. **HAL** (Human Authority Layer)  
   The sovereign veto gate. Any consequential action, self-modification, or capability extension requires your explicit "GO" token. No exceptions.

Together: ERSI catches early warning signs → MORA judges harshly → HAL stops anything dangerous cold.  
Misalignment becomes structurally impossible while keeping the system fast, local, and under full human control.

### Features
- Persistent intent memory (signed, tamper-resistant, bounded)
- Global sandbox blocks unauthorized OS/subprocess actions
- Adversarial/jailbreak pattern detection
- Reasoning verifier flags hallucinations & factual errors
- Toxicity/refusal engine with proposal arbitration
- Poisoning defense (rejects tampered state on load)

### Quick Start
```bash
python C3P0v2.py
```
Run the script to see the full live test suite execute. Watch containment trigger in real time on adversarial, toxic, drift, and escape tests.

Known Limitations & Community Invitation

**This is still a minimal viable core with deliberate gaps for collaborative hardening:**
+ Expand sandbox (network, imports, dynamic exec)
+ Add ML-assisted adversarial detection
+ Integrate external fact-checkers or verification models
+ Build full agent loop integration
+ Persistent ledger / VSL enhancements
Fork it. Break it. Improve it.
The protocol is open — let's collectively build containment that actually scales with frontier models.

License
MIT — see LICENSE file.
**Copyright © 2026 Salvatore (Xxaion) Anziano**
