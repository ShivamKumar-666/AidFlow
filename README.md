# AidFlow — Multi-Domain Essential Goods Redistribution Platform
### Autonomous, Multi-Agent Food & Medicine Redistribution Orchestration System

**Status: Production-Ready ✅ (91/91 Tests Passing, CI Integrated)**

**AidFlow** is a web-based, agentic platform that extends food waste redistribution into a broader essential-goods ecosystem — adding pharmaceutical viability modeling and medicine verification agents alongside the existing food rescue pipeline.

The system automates the routing of surplus goods (food and medicine) from donors (restaurants, grocery stores, pharmacies, clinics) to NGOs and relief organizations — reducing waste while maximizing social impact.

---

## 🌟 Key Features

* **Multimodal GenAI Intake:** Donors submit listings via photo (Vision Intake using Groq Llama Vision) or conversational description (Chat Intake using Groq Llama 3 70B). The system extracts structured fields automatically with donor verification.
* **XGBoost Freshness Prediction:** Predicts freshness scores for food donations based on temporal parameters and sensory details (smell, texture, moisture).
* **Medicine Viability Modeling:** Extends the ML layer with pharmaceutical shelf-life and stability modeling for medicine donations — original to AidFlow.
* **Hybrid Medicine Verification Agent:** A new LangGraph agent node that cross-references medicine metadata (batch, expiry, regulatory status) against structured knowledge — original to AidFlow.
* **SHAP Explainability:** Surfaces feature contributions explaining *why* the model predicted a freshness/viability score, translated into natural language by an LLM for NGOs.
* **Dual RAG Schema:** Separate Qdrant collections for food NGO profiles and medicine-capable NGO profiles with domain-specific weighted matching — extended for AidFlow.
* **LangGraph Multi-Agent Orchestration:** Runs an autonomous pipeline of specialized agents (Intake, Verification, Matching, Logistics) with conditional routing and self-loop escalations.
* **Observability Dashboard:** Real-time monitoring dashboard displaying pipeline status distributions, agent performance charts, and step-by-step decision trails.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      OBSERVABILITY DASHBOARD                    │
│           (Django Templates + Chart.js Pipeline Auditing)       │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│             LANGGRAPH MULTI-AGENT STATE MACHINE                 │
│  [Intake] ──► [Verify] ──► [Medicine Verify*] ──► [Match] ──► [Logistics]  │
└───────┬────────────────────┬─────────────┬─────────────▲────────┘
        │                    │             │             │
┌───────▼───────┐    ┌───────▼───────┐     │     ┌───────┴────────┐
│   ML LAYER    │    │  GENAI LAYER  │     │     │   RAG LAYER    │
│  XGBoost +    │    │ Llama Vision  │     │     │  Dual Schema*  │
│  SHAP Engine  │    │ + JSON Mode   │     │     │   + Qdrant     │
│  Med Viability*│   │               │     │     │                │
└───────────────┘    └───────────────┘     │     └────────────────┘
                                           │
┌──────────────────────────────────────────▼──────────────────────┐
│                  DJANGO REST API + POSTGRESQL                   │
│          (Data Persistence, User Auth, and API Routing)         │
└─────────────────────────────────────────────────────────────────┘

  * = Original to AidFlow (not present in EcoFeast)
```

---

## 📁 Repository Structure

```bash
AidFlow/
├── config/             # Django project configuration
├── donations/          # Core donation models, serializers, and views
├── users/              # Custom user roles (Donors/NGOs) and capability profiles
├── ml_service/         # Feature engineering, XGBoost training, and SHAP explainability
│   └── medicine/       # ★ Medicine viability modeling (original to AidFlow)
├── genai_service/      # LLM-based image/text information extraction
├── rag_service/        # Qdrant NGO profile embeddings and weighted matching (dual schema)
├── agents/             # LangGraph state definition and agent node handlers
│   └── medicine_verification_agent.py  # ★ Original to AidFlow
├── templates/          # HTML templates (Dashboard, Maps, and Agent Monitoring)
├── tests/              # Pytest test suite for ML, GenAI, RAG, and Agents
├── Dockerfile          # Production web service dockerfile
└── docker-compose.yml  # Local services (Django, Postgres, Redis, Qdrant)
```

---

## 🚀 Getting Started

### Prerequisites
* Docker and Docker Compose
* Python 3.10+ (if running locally without Docker)
* A [Groq API Key](https://console.groq.com/) for GenAI features

### Running with Docker (Recommended)

1. **Clone the repository and enter the directory:**
   ```bash
   git clone https://github.com/ShivamKumar-666/AidFlow.git
   cd AidFlow
   ```

2. **Create a `.env` file in the project root:**
   ```env
   DEBUG=1
   SECRET_KEY=your_django_secret_key
   GROQ_API_KEY=your_groq_api_key
   POSTGRES_DB=aidflow
   POSTGRES_USER=aidflow
   POSTGRES_PASSWORD=aidflow_dev_2026
   QDRANT_URL=http://qdrant:6333
   ```

3. **Build and start the container services:**
   ```bash
   docker-compose up --build
   ```
   This will spin up:
   * **Django Web Server** at `http://localhost:8000`
   * **PostgreSQL Database** at `localhost:5432`
   * **Redis Cache/Broker** at `localhost:6379`
   * **Qdrant Vector DB** at `localhost:6333`

4. **Run migrations and populate mock data:**
   ```bash
   docker-compose exec web python manage.py migrate
   # Optional: Sync RAG profiles (food + medicine NGOs)
   docker-compose exec web python manage.py shell -c "from rag_service.matcher import sync_all_ngos; sync_all_ngos()"
   ```

---

## 🧪 Verification & Testing

The project uses `pytest` for unit and integration testing.

Run all tests inside the Docker container:
```bash
docker-compose exec web pytest
```

The test suite covers:
* **Models:** CustomUser profiles and Donation constraints.
* **ML Service:** Feature transform pipelines, model predictions, and SHAP explainer runs.
* **Medicine ML:** Viability modeling and pharmaceutical shelf-life predictions.
* **GenAI Service:** Llama extraction correctness and explainers (mocked API).
* **RAG Service:** Qdrant upserts and combined distance/capacity matcher (dual schema).
* **Agents:** LangGraph StateGraph state updates, validation filters, and routing loops (including Medicine Verification Agent).

---

## Attribution

AidFlow extends the architecture of [EcoFeast](https://github.com/ShivamKumar-666/EcoFeast/tree/shivam),
a group final year project. The food redistribution layer, Django foundation, and LangGraph
orchestration pattern originate there. Everything in `ml_service/medicine/`,
`agents/medicine_verification_agent.py`, and the dual RAG schema is original to AidFlow.
