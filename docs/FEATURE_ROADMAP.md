# JD Generator: Feature Roadmap

**Project**: LLM Job Description Generator - Archetypes Pipeline  
**Audience**: Engineering & Product  
**Owner**: R&D Lead  

---

## Target State Overview

The system enables high-quality JD generation by leveraging learned "archetypes" - representative patterns extracted from historical JD data. The pipeline has two major parts:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ARCHETYPE CREATION                               │
│  Historical JDs → Extract → Cluster → Aggregate → Name → Archetypes    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                            [Archetype Store]
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        JD GENERATION (R&D Pending)                      │
│  User Input → Match Archetype → Retrieve Context → Generate Draft JD   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Archetype Creation Pipeline

### Target State

A batch pipeline that processes the historical JD corpus and produces a set of named archetypes, each containing aggregated requirements, skills, and patterns for a "type" of role.

---

### Feature 1.1: Structured Extraction

**What it does**: LLM extracts structured data (skills, qualifications, experience, licenses, etc.) from unstructured JD text.

| R&D Milestones | Product Inputs Needed |
|----------------|----------------------|
| Define extraction schema (what fields to extract) | Confirm which requirement types matter for JD quality |
| Validate extraction accuracy on sample corpus (target: X% precision/recall) | Provide labeled examples for validation |
| Test schema on edge cases (short JDs, non-standard formats) | Identify known problematic JD formats in corpus |

**Infra Required**: LLM model (chat/completion)

**Current Status**: Code complete, validation pending on full dataset

---

### Feature 1.2: Embedding & Clustering

**What it does**: Creates vector representations of extracted data, then clusters similar JDs into groups that become archetype candidates.

| R&D Milestones | Product Inputs Needed |
|----------------|----------------------|
| Determine optimal embedding approach (skills-only vs contextual vs structured) | Define what "similar roles" means from business perspective |
| Tune clustering parameters for meaningful archetype granularity | Review cluster outputs - are these sensible role groupings? |
| Establish cluster quality metrics (silhouette score, purity against existing taxonomies) | Provide existing role family / job family mappings for comparison |

**Infra Required**: Embedding model, Vector DB (for storage/retrieval)

**Current Status**: Code complete, parameter tuning in progress

---

### Feature 1.3: Aggregation & Naming

**What it does**: Within each cluster, aggregates common requirements (with frequency/confidence) and generates a meaningful archetype name.

| R&D Milestones | Product Inputs Needed |
|----------------|----------------------|
| Define aggregation logic (frequency thresholds, confidence scoring) | Define what "required" vs "preferred" means for aggregated skills |
| Validate aggregated archetypes represent the cluster well | Review sample archetypes - do they match intuition? |
| LLM naming produces clear, consistent labels | Provide naming conventions / terminology preferences |

**Infra Required**: LLM model (chat/completion)

**Current Status**: Code complete, quality validation pending

---

### Feature 1.4: Archetype Storage

**What it does**: Persists archetypes in a queryable format for downstream consumption (generation, retrieval, analysis).

| R&D Milestones | Product Inputs Needed |
|----------------|----------------------|
| Define archetype schema (what gets stored) | Confirm what metadata should be attached (org mappings, usage stats, etc.) |
| Design retrieval patterns (by embedding similarity, by metadata filters) | Define how users/systems will query archetypes |

**Infra Required**: NoSQL DB or document store, Vector DB (for similarity search)

**Current Status**: Schema design pending

---

### Feature 1.5: Archetype Maintenance (Post-MVP)

**What it does**: Keeps archetypes fresh as new JDs are created and role patterns evolve.

| R&D Milestones | Product Inputs Needed |
|----------------|----------------------|
| Define refresh strategy (full rebuild vs incremental update) | How often should archetypes be refreshed? |
| Detect archetype drift (new role types emerging, existing ones shifting) | Process for reviewing/approving archetype changes |
| Handle archetype versioning | Governance model for archetype lifecycle |

**Infra Required**: Batch orchestration / scheduling, change detection pipeline

**Current Status**: Deferred to post-MVP

---

## Part 2: JD Generation from Archetypes

### Target State

Given user inputs (role type, level, team context, specific requirements), the system retrieves relevant archetype(s) and generates a draft JD.

**Status: R&D Pending** - Core generation approach not yet validated.

---

### Feature 2.1: Archetype Matching

**What it does**: Given user input, find the most relevant archetype(s) to use as foundation.

| R&D Milestones | Product Inputs Needed |
|----------------|----------------------|
| Define matching approach (embedding similarity? metadata rules? hybrid?) | What inputs will users provide? (structured form vs free text) |
| Handle "no good match" cases | Fallback behavior when no archetype fits |
| Test matching accuracy against user intent | Provide test cases with expected archetype matches |

**Unknowns**:
- How much user input is needed to reliably match?
- Single archetype vs blending multiple archetypes?

**Infra Required**: Vector DB (for similarity search), retrieval API

---

### Feature 2.2: Context Retrieval

**What it does**: Beyond the archetype, retrieve additional context to personalize the JD (team descriptions, similar recent JDs, org-specific language).

| R&D Milestones | Product Inputs Needed |
|----------------|----------------------|
| Determine what context improves generation quality | What org-specific context exists? (team descriptions, style guides, etc.) |
| RAG approach vs direct context injection | Access to supplementary data sources |
| Test context window limits / chunking strategies | Prioritization: what context matters most? |

**Unknowns**:
- What context sources are available and accessible?
- How much context is "enough" vs diminishing returns?

**Infra Required**: Vector DB, document retrieval pipeline

---

### Feature 2.3: Draft Generation

**What it does**: LLM generates a complete JD draft using archetype + context + user inputs.

| R&D Milestones | Product Inputs Needed |
|----------------|----------------------|
| Prompt engineering for consistent, high-quality output | Define quality criteria for generated JDs |
| Test output against existing "gold standard" JDs | Provide gold standard examples for comparison |
| Handle structured sections (responsibilities, qualifications, etc.) | Required JD structure / template |
| Ensure compliance language is included correctly | Compliance requirements by role type / location |

**Unknowns**:
- Best generation strategy: section-by-section vs full document?
- How to handle conflicting requirements (archetype vs user input)?
- Edit distance from generated draft to final JD (quality proxy)

**Infra Required**: LLM model (chat/completion)

---

### Feature 2.4: User Editing & Feedback (Post-MVP)

**What it does**: Capture user edits to generated JDs and feed back into system improvement.

| R&D Milestones | Product Inputs Needed |
|----------------|----------------------|
| Define feedback signal (what edits indicate quality issues) | UX for capturing structured feedback |
| Close the loop: edits → archetype refinement | Process for incorporating feedback |

**Unknowns**:
- How to attribute quality issues (archetype problem vs generation problem vs user preference)?

**Infra Required**: Feedback storage, analytics pipeline

**Current Status**: Deferred to post-MVP

---

## MVP Scope

### What's In

| Feature | MVP Approach |
|---------|-------------|
| Structured Extraction | Full implementation |
| Embedding & Clustering | Full implementation |
| Aggregation & Naming | Full implementation |
| Archetype Storage | Static export (JSON/DB dump from pipeline run) |
| Archetype Matching | Basic implementation (TBD based on R&D) |
| Draft Generation | Basic implementation (TBD based on R&D) |

### What's Deferred

| Feature | Reason |
|---------|--------|
| Archetype Maintenance | Not needed until archetypes go stale; adds complexity |
| Dynamic Refresh | MVP uses static archetypes from initial pipeline run |
| User Feedback Loop | Requires production usage data first |
| Advanced Context Retrieval | Start simple, add sophistication based on quality gaps |

---

## MVP to Target: Enhancement Path

```
MVP (Static Archetypes)
    │
    ├── Enhancement 1: Scheduled Refresh
    │   Add batch job to periodically rebuild archetypes
    │
    ├── Enhancement 2: Incremental Updates  
    │   New JDs trigger archetype updates without full rebuild
    │
    ├── Enhancement 3: Context Enrichment
    │   Add team descriptions, org context to generation
    │
    └── Enhancement 4: Feedback Integration
        User edits inform archetype quality scoring
```

---

## Infrastructure Summary

| Component | Purpose |
|-----------|---------|
| LLM Model (chat/completion) | Extraction, naming, generation |
| Embedding Model | Create vector representations of JD content |
| Vector DB | Store embeddings, enable similarity search |
| NoSQL / Document Store | Store archetypes, JD metadata, pipeline outputs |
| Batch Orchestration | Run archetype pipeline on schedule (post-MVP) |

---

## Open Questions for Product

1. **Extraction Schema**: What requirement types must we capture? (skills, certifications, experience levels, education, etc.)

2. **Archetype Granularity**: How specific should archetypes be? (e.g., "Software Engineer" vs "Senior Backend Engineer - Payments")

3. **User Input Model**: What will users provide when requesting a JD? (structured form fields? free text description? existing JD to iterate on?)

4. **Quality Definition**: What makes a generated JD "good enough"? (measurable criteria for validation)

5. **Existing Taxonomies**: Do role family codes / job family mappings exist that archetypes should align to?

6. **Refresh Cadence**: How often do role patterns change enough to warrant archetype refresh?

---

## R&D Priority: Generation Unknowns

The following require focused R&D before committing to generation approach:

| Unknown | Experiment Needed |
|---------|------------------|
| Matching reliability | Test: given N user inputs, how often do we match the "right" archetype? |
| Single vs multi-archetype | Test: does blending archetypes improve or hurt quality? |
| Context value | Test: A/B generation with/without additional context - quality delta? |
| Generation strategy | Test: full-doc generation vs section-by-section - quality/consistency tradeoff? |
| Prompt sensitivity | Test: how much does output vary with prompt changes? |

---

*Document Version: 1.0*  
*Last Updated: December 2024*
