# build-your-own-x — Vibe-Coding Edition

> *What I cannot create, I do not understand — Richard Feynman.* 
>  
> Use **Vibe-Coding** to make “from-scratch” building faster, smoother, and more fun.

This repo preserves the spirit of **build-your-own-x**, and upgrades the workflow for the **LLM era**:
- **Vibe-Coding** = short iterative loops + lightweight prompt recipes + runnable scaffolds + tiny wins.
- **LLMs in the loop**: prompt recipes, debug diaries, and auto-checks live alongside the code.

## What is Vibe-Coding?
A pragmatic build style:
1. **Micro loops (15–25 min)**: set one tiny goal → build → run → jot what felt off (the “vibe”).  
2. **Prompt recipes**: reusable 10–15 line prompts for planning, refactors, tests, and perf hints.  
3. **Scaffolds over specs**: start from minimal templates; ship something runnable ASAP.  
4. **Reality checks**: each step ends with an *auto-runnable check* (script/notebook/test).  
5. **Build diary**: keep a `vibes.md` per project — 1–3 bullets/loop: intent → friction → next.

> TL;DR: Less ceremony, more momentum. Learn a bit, run a bit, close the loop.

## To-D List: 

We’re actively collecting:
**Community showcases** — repos/blog posts that used **Vibe-Coding** to complete a BYO-X project.  

### How to submit
Open a PR or issue with:
- **Link(s):** repo and/or blog post
- **Vibe artifacts:** `recipes/*.md` (prompts), `checks/*` (auto-checks), `scaffolds/*` (minimal runnable)
- **Build diary:** `vibes.md` (3 bullets: intent → friction → next)
- **Notes:** compute used, dataset/source, license

---

### 1) Community Vibe-Coding Showcases (repos/blogs)

> Looking for real build diaries + prompts that wrap classic BYO-X topics.

- [ ] **3D Renderer** — repo/blog + prompts + scaffold + check  
- [ ] **Augmented Reality** — repo/blog + prompts + scaffold + check  
- [ ] **BitTorrent Client** — repo/blog + prompts + scaffold + check  
- [ ] **Blockchain / Cryptocurrency** — repo/blog + prompts + scaffold + check  
- [ ] **Bot (Discord/Slack/etc.)** — repo/blog + prompts + scaffold + check  
- [ ] **Command-Line Tool** — repo/blog + prompts + scaffold + check  
- [ ] **Database (mini-KV / Redis-like)** — repo/blog + prompts + scaffold + check  
- [ ] **Docker / Container-from-scratch** — repo/blog + prompts + scaffold + check  
- [ ] **Emulator / VM** — repo/blog + prompts + scaffold + check  
- [ ] **Front-end Framework / React-from-scratch** — repo/blog + prompts + scaffold + check  
- [ ] **Game (Tetris/Rogue/etc.)** — repo/blog + prompts + scaffold + check  
- [ ] **Git internals (mini-git)** — repo/blog + prompts + scaffold + check  
- [ ] **Network Stack** — repo/blog + prompts + scaffold + check  
- [ ] **Operating System (toy kernel/bootloader)** — repo/blog + prompts + scaffold + check  
- [ ] **Physics Engine** — repo/blog + prompts + scaffold + check  
- [ ] **Programming Language / Compiler** — repo/blog + prompts + scaffold + check  
- [ ] **Regex Engine** — repo/blog + prompts + scaffold + check  
- [ ] **Search Engine** — repo/blog + prompts + scaffold + check  
- [ ] **Shell** — repo/blog + prompts + scaffold + check  
- [ ] **Template Engine** — repo/blog + prompts + scaffold + check  
- [ ] **Text Editor** — repo/blog + prompts + scaffold + check  
- [ ] **Visual Recognition System** — repo/blog + prompts + scaffold + check  
- [ ] **Voxel Engine** — repo/blog + prompts + scaffold + check  
- [ ] **Web Browser** — repo/blog + prompts + scaffold + check  
- [ ] **Web Server / Framework** — repo/blog + prompts + scaffold + check  
- [ ] **Uncategorized (surprise us!)** — repo/blog + prompts + scaffold + check

---

### 2) Mini-GPT with Vibe-Coding 

> Concrete, runnable examples that demonstrate the Vibe loop on DL projects.

#### Core LLM / NLP
- [ ] **Mini-GPT (toy)** — tokenizer + tiny Transformer + sampling  


## Repo Layout (recommended)


## How to Use
1. Pick a topic → open `blueprints/<topic>/README.md`.  
2. Copy the **Prompt Recipes** and follow the steps (each section stands alone).  
3. Run the **auto-check** via `checks/<topic>_smoke.*` or a small notebook.  
4. Log your loop in `vibes.md` so you can reproduce & share later.

### The Vibe Loop (cheat-sheet)
- 🎯 **Tiny intent** (≤25 min): define one win.  
- ✍️ **Plan w/ recipe**: paste a 10–15 line prompt (plan/refactor/test).  
- 🛠️ **Do the thing**: code the minimum; prefer a scaffold.  
- ✅ **Auto-check**: run `checks/*` and capture the output.  
- 📝 **Log vibe**: append 1–3 bullets to `vibes.md` (intent → friction → next).





## Table of Contents
- [build-your-own-x — Vibe-Coding Edition](#build-your-own-x--vibe-coding-edition)
  - [What is Vibe-Coding?](#what-is-vibe-coding)
  - [To-D List:](#to-d-list)
    - [How to submit](#how-to-submit)
    - [1) Community Vibe-Coding Showcases (repos/blogs)](#1-community-vibe-coding-showcases-reposblogs)
    - [2) Mini-GPT with Vibe-Coding](#2-mini-gpt-with-vibe-coding)
      - [Core LLM / NLP](#core-llm--nlp)
  - [Repo Layout (recommended)](#repo-layout-recommended)
  - [How to Use](#how-to-use)
    - [The Vibe Loop (cheat-sheet)](#the-vibe-loop-cheat-sheet)
  - [Table of Contents](#table-of-contents)
  - [Build your own `LLM / Agentic System`](#build-your-own-llm--agentic-system)
    - [Build your own `LLM (from scratch / minimal)`](#build-your-own-llm-from-scratch--minimal)
    - [Build your own `RAG Pipeline`](#build-your-own-rag-pipeline)
    - [Build your own `Agent`](#build-your-own-agent)
    - [Build your own `Evaluation & Benchmark`](#build-your-own-evaluation--benchmark)
    - [Build your own `Safety Guard`](#build-your-own-safety-guard)
    - [Build your own `Vision / Multimodal`](#build-your-own-vision--multimodal)
    - [Build your own `Inference & Deployment`](#build-your-own-inference--deployment)
  - [Contribute](#contribute)
  - [Credits](#credits)

## Build your own `LLM / Agentic System`

> Hands-on blueprints with **LLMs in the loop** (planning prompts + runnable checks + tiny scaffolds).

### Build your own `LLM (from scratch / minimal)`
* [**Python**] _Mini-GPT (Toy)_: tokenizer → tiny Transformer → next-token sampling.  
  **Vibe add-on**: `recipes/llm_planning.md`, `checks/llm_smoke.py`, `scaffolds/llm_tiny/`

### Build your own `RAG Pipeline`
* [**Python**] _RAG Minimal_: loader → chunker → embed → vector store → retriever → rerank → answer.  
  **Vibe add-on**: `checks/rag_eval.py` (retrieval quality), query-rewrite & grading prompt recipes.

### Build your own `Agent`
* [**Python**] _Single-tool Agent_: ReAct-style reasoning + one tool (web or bash).  
  **Vibe add-on**: `recipes/agent_loop.md` (CoT, stop conditions, failure fallback), `checks/agent_replay.py`

### Build your own `Evaluation & Benchmark`
* [**Python**] _Judge-Pair Eval_: pairwise judge + majority vote + rubric prompts.  
  **Vibe add-on**: `checks/judge_consistency.py` (retries & agreement), `scaffolds/judge_pair/`

### Build your own `Safety Guard`
* [**Python**] _Prompt-Injection Filter_: pattern rules + LLM referee + allowlist.  
  **Vibe add-on**: `recipes/safety_redteam.md`, `checks/prompt_injection_suite.py`

### Build your own `Vision / Multimodal`
* [**Python**] _Mini-VLM Demo_: image encoder + text decoder + projection + caption/QA.  
  **Vibe add-on**: `checks/vlm_caption_bleu.py`, tiny sample data + simple visualizations.

### Build your own `Inference & Deployment`
* [**Docker/Bash**] _Tiny Serving Stack_: FastAPI + streaming + batching + metrics + autoscaling hints.  
  **Vibe add-on**: `scaffolds/serving_tiny/`, `checks/latency_p99.sh`

## Contribute

We love classic “from-scratch” tutorials *and* Vibe-wrapped upgrades.

**For each PR, please include:**
1) One minimal runnable example (script/notebook/app)  
2) A 10–15 line **prompt recipe** (planning/refactor/test/perf) in `recipes/`  
3) An **auto-check** in `checks/` (smoke test or small eval)  
4) 3 short bullets from your `vibes.md` (intent → friction → next)

You can also “Vibe-wrap” any existing entry by adding a tail line under it:


Submissions welcome — open a PR or [create an issue](https://github.com/codecrafters-io/build-your-own-x/issues/new).  
Help review [pending submissions](https://github.com/codecrafters-io/build-your-own-x/issues) with comments and reactions.

## Credits

This project is inspired by and extends the free, community-maintained repository **Build Your Own X**.

- Original repo: **[codecrafters-io/build-your-own-x](https://github.com/codecrafters-io/build-your-own-x)**
- Started by **[Daniel Stefanovic](https://github.com/danistefanovic)**, now maintained by **[CodeCrafters, Inc.](https://codecrafters.io)**
- License: **[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)** (rights waived to the extent possible under law)

*Not affiliated with CodeCrafters; we simply ❤️ their work and the community around it.*

