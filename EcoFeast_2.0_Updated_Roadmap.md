# EcoFeast 2.0 — Updated Roadmap: ML → GenAI → Agentic AI
### Final Year Group Project | Food Redistribution Only

**Goal:** Evolve EcoFeast from a "form + Random Forest model" Django app into an autonomous, multi-agent food-rescue orchestration system — a final year project that demonstrates classical ML, applied GenAI, and agentic AI in one cohesive, demo-able product.

---

## 0. Current State (Baseline)

| Layer | What Exists | Files |
|---|---|---|
| Backend | Django app, REST endpoints | `config/`, `donations/`, `users/` |
| ML | Random Forest freshness classifier inside donations app | `donations/ml_model.py` |
| Data | Sensory CSV dataset (texture, smell, moisture, temp, etc.) | `donations/food_data.csv` |
| Frontend | Django templates + Leaflet.js maps | `templates/` |
| Database | SQLite (dev) | — |
| External APIs | Open-Meteo (weather), Nominatim (geocoding) | — |

**Critical gaps to fix before building anything new:**
- ML code is embedded inside the Django app — no separation of concerns
- Two freshness implementations exist (rule-based fallback + Random Forest) — unify these
- SQLite in dev means bugs that would crash Postgres pass silently
- No LLM/GenAI layer
- No autonomy — every step is human-triggered (donor fills form → NGO manually browses map and claims)

---

## Target Architecture — 3 Layers

```
┌─────────────────────────────────────────────┐
│  AGENTIC LAYER (LangGraph orchestrator)      │
│  Intake → Verification → Matching →          │
│  Logistics → Escalation                      │
├─────────────────────────────────────────────┤
│  GENAI LAYER                                 │
│  Vision intake · Conversational assistant ·  │
│  RAG over NGO profiles · Explanation gen     │
├─────────────────────────────────────────────┤
│  ML LAYER                                    │
│  Freshness (XGBoost unified + SHAP)          │
└─────────────────────────────────────────────┘
        Django REST API + PostgreSQL
        Redis + Celery (async tasks)
```

---

## Tech Stack Changes from Current → 2.0

| Component | Current | Updated | Why |
|---|---|---|---|
| ML model | Random Forest (Scikit-Learn) | XGBoost + SHAP | Better accuracy, feature explainability for interviews |
| ML location | `donations/ml_model.py` | `ml_service/` module | Separation of concerns, no train/serve skew |
| Database | SQLite | PostgreSQL (Docker) | ACID compliance, FK enforcement, production-ready |
| Task queue | None | Celery + Redis | Async NGO notifications + logistics timeout re-routing |
| Vector store | None | Qdrant | NGO profile RAG matching with filtered search |
| Embeddings | None | HuggingFace `all-MiniLM-L6-v2` | Free, proven, already used in similar projects |
| LLM | None | Groq Llama 3.3 70B + vision model | Free tier, fast inference, function-calling support |
| Orchestration | None | LangGraph | Multi-agent state machine with autonomous re-routing |
| Frontend | Django templates | Django templates + React dashboard | Keep templates for forms, React for live agent dashboard |
| DevOps | None | Docker Compose + GitHub Actions CI | Portfolio credibility, pytest-verified |

---

## Phase 1 — Fix the Foundation (Week 1)

**Do this before anything else. Everything downstream depends on it.**

### 1a. Switch to PostgreSQL
- Set up Docker Compose with Postgres locally
- Migrate from SQLite — run `makemigrations` + `migrate` on clean Postgres instance
- Update `settings.py` to use `dj-database-url` for environment-based DB config
- Never commit `.env` with DB credentials

### 1b. Refactor ML out of the Django app
- Create `ml_service/` at project root — completely separate from `donations/`
- Move and rewrite `donations/ml_model.py` → `ml_service/trainer.py` + `ml_service/predictor.py`
- Create one `FeatureBuilder` class used by both trainer and predictor — eliminates train/serve skew
- Deprecate the rule-based fallback as a documented last-resort only (model fails to load → fallback, not default path)

### 1c. Upgrade Random Forest → XGBoost + SHAP
- Retrain on the existing food CSV with XGBoost
- Add `ml_service/explainer.py` — for every prediction return top 3 SHAP feature contributions
  - Example output: `{"storage_time": -0.18, "smell_score": -0.12, "container_type": +0.05}`
- Add `ml_service/evaluate.py` — precision/recall per class, confusion matrix, ROC-AUC
- Version the model file: `ml_service/models/freshness_xgb_v1.json`
- **Quote real numbers in your project report and interviews — not "it works"**

**New/changed files:**
```
ml_service/
├── __init__.py
├── feature_builder.py
├── trainer.py
├── predictor.py
├── explainer.py
├── evaluate.py
└── models/
    └── freshness_xgb_v1.json
```

---

## Phase 2 — GenAI Intake Layer (Weeks 2–3)

### 2a. Vision Intake
- Donor uploads a food photo instead of filling 8 dropdowns manually
- Call Groq's vision-capable Llama model to extract:
  - `food_type`, `container_type`, `estimated_quantity_kg`, visual freshness cues
- Output auto-populates the donation form — donor confirms/edits before submit
- **Human-in-the-loop checkpoint is mandatory here** — do not auto-submit from vision output alone
- Fallback: if vision confidence is low, show the manual form

### 2b. Conversational Donor Assistant
- Free-text input: *"Leftover biryani from a wedding, cooked 4 hrs ago, steel containers, ~15kg"*
- Groq Llama 3.3 70B does structured extraction into `Donation` model fields via JSON mode / function calling
- Fallback to manual form if extraction confidence is below threshold
- Keeps the UX fast for repeat donors who know what they're submitting

### 2c. SHAP-to-Language Explanation
- Take SHAP output from Phase 1 and pass it to the LLM
- Generate one human-readable line shown to NGOs on the map dashboard
- Example: *"Fresh — minimal time decay, refrigerated storage, low-risk food category"*
- NGOs make faster, more confident pickup decisions with this context

**New files:**
```
genai_service/
├── __init__.py
├── vision_intake.py
├── chat_intake.py
└── explainer_llm.py
```

---

## Phase 3 — RAG-Based NGO Matching (Week 4)

**Replaces "nearest NGO on map" with "best-fit NGO given real constraints."**

- Each NGO gets a free-text capability document:
  - Dietary restrictions they serve, cultural/religious rules, storage capacity, operating hours, past reliability score
- Embed all NGO profiles with HuggingFace `all-MiniLM-L6-v2` into Qdrant
- Matching query = donation metadata (food type, quantity, freshness urgency) → retrieve top-k compatible NGOs by semantic similarity
- Re-rank results by: geographic distance (haversine) + current capacity + reliability score
- This is the architectural upgrade that makes matching non-trivial and worth explaining

**New files:**
```
rag_service/
├── __init__.py
├── ngo_embeddings.py
└── matcher.py
```

---

## Phase 4 — Agentic Orchestration (Weeks 5–6)

**This is the core differentiator. Build it right.**

### Agent Responsibilities

| Agent | What It Does |
|---|---|
| **Intake Agent** | Normalizes donor input (text / image / manual form), invokes freshness model, returns structured `DonationState` |
| **Verification Agent** | Plausibility checks — flags anomalies in quantity/time claims, donor history outliers, implausible food/weather combinations |
| **Matching Agent** | Calls RAG matcher (Phase 3), produces ranked NGO list with scores |
| **Logistics Agent** | Sends claim offer to top NGO, starts 10-min timeout; if no response → autonomously escalates to next-ranked NGO without human intervention |
| **Orchestrator** | LangGraph state machine wiring the above; persists full agent decision trail to Postgres for auditability |

### The Key Autonomy Behavior
The Logistics Agent must re-route **without a human clicking anything**. This is the demonstrable proof point for your viva and for interviews. Without this, it's ML + a chatbot, not an agent.

### LangGraph Skeleton

```python
from langgraph.graph import StateGraph, END
from agents import intake_agent, verification_agent, matching_agent, logistics_agent

graph = StateGraph(DonationState)

graph.add_node("intake", intake_agent)
graph.add_node("verify", verification_agent)
graph.add_node("match", matching_agent)
graph.add_node("logistics", logistics_agent)

graph.set_entry_point("intake")
graph.add_edge("intake", "verify")
graph.add_conditional_edges(
    "verify",
    lambda s: "match" if s.is_valid else END
)
graph.add_edge("match", "logistics")
graph.add_conditional_edges(
    "logistics",
    lambda s: "match" if s.needs_escalation else END  # autonomous re-route
)

app = graph.compile()
```

### Agent State Schema

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DonationState:
    donation_id: str
    food_type: str
    quantity_kg: float
    freshness_score: float
    shap_explanation: dict
    is_valid: bool
    matched_ngos: List[dict]
    assigned_ngo: Optional[str]
    needs_escalation: bool
    escalation_count: int
    decision_trail: List[dict]  # persisted to Postgres
```

**New files:**
```
agents/
├── __init__.py
├── intake_agent.py
├── verification_agent.py
├── matching_agent.py
├── logistics_agent.py
└── orchestrator.py
```

---

## Phase 5 — Observability Dashboard (Week 7)

**This is what you screen-share in your viva and demo videos. Make it count.**

- React-based dashboard (separate from Django templates)
- Shows per-donation: agent decision trail, escalation events, SHAP explanation, NGO match scores
- Live freshness score distribution across active donations
- NGO response time analytics
- Connects to Django REST API via axios

**Why this matters:** it makes the "agentic" claim tangible. Without a visible decision trail, you're asking evaluators to take your word for it.

---

## Phase 6 — DevOps + Polish (Week 8)

- Docker Compose: Django + Postgres + Redis + Qdrant in one `docker-compose.yml`
- GitHub Actions CI: run pytest on every push, fail the build if tests break
- Minimum 50 pytest-verified tests covering: ML predictor, agent state transitions, RAG matcher, API endpoints
- Environment-based config: `.env.example` committed, `.env` gitignored
- Production deployment: Render (Django) + managed Postgres

---

## What's Deliberately Excluded

| Feature | Reason Excluded |
|---|---|
| Demand forecasting model | No historical claims data — model would train on synthetic data, meaningless |
| Medicine/clothing/other goods | Out of scope for EcoFeast — reserved for AidFlow fork |
| Voice input | Adds complexity without proportional demo value for group project |
| Mobile app | Out of scope |

---

## Suggested Week-by-Week Timeline

| Week | Focus | Demo-able After? |
|---|---|---|
| 1 | Postgres + ML refactor (XGBoost + SHAP) | Yes — show before/after accuracy + SHAP output |
| 2–3 | GenAI intake (vision + conversational + explanations) | Yes — live photo → form population demo |
| 4 | RAG NGO matching | Yes — show semantic match vs nearest-only match |
| 5–6 | LangGraph agentic orchestration | Yes — autonomous re-routing demo |
| 7 | Observability dashboard | Yes — full decision trail visualization |
| 8 | Docker + CI + tests + deployment polish | Yes — live URL to share |

Each phase is independently demo-able. You can stop after Phase 4 and still defend a strong "ML + GenAI + Agentic AI" narrative at your viva.

---

## Risks to Watch

| Risk | Mitigation |
|---|---|
| Groq free tier rate limits under demo load | Cache repeated NGO-profile RAG queries in Redis |
| Vision intake auto-submitting bad data | Mandatory donor confirmation step before any agent proceeds |
| LangGraph orchestrator not visibly "doing anything" | Observability dashboard (Phase 5) is non-optional for viva |
| Eval numbers not quoted | Keep a labeled eval set, run `evaluate.py` and record precision/recall/ROC-AUC before viva |
| Group contribution imbalance | Assign phases to individuals clearly, commit under personal GitHub accounts |

---

## Interview / Viva Narrative

- **ML:** *"Replaced a Random Forest baseline with XGBoost, added SHAP-based explainability that surfaces the top 3 contributing features per prediction with real precision/recall metrics."*
- **GenAI:** *"Added multimodal intake — donors submit via photo or natural language, extracted into structured data via LLM function-calling with a mandatory human confirmation checkpoint."*
- **Agentic AI:** *"Designed a LangGraph multi-agent pipeline that autonomously verifies donations, semantically matches NGOs by capability, and re-routes unclaimed donations without human intervention — with a full persisted decision trail for auditability."*

The last line is what differentiates you from "I used an LLM API" candidates.


