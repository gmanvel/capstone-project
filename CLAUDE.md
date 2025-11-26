# HR Chatbot - Claude Code Guide

## Project Overview

Multi-agent RAG system for HR queries using Confluence as knowledge base.

**Architecture**: 2 deployable units (API service + Background job) with shared RAG pipeline module.

## Critical Instructions

### Before Starting ANY Task

**MANDATORY**: Read these architecture documents first:

1. `architecture/ARCHITECTURE.md` - System design, data flow, tech stack
2. `architecture/CODING_STANDARDS.md` - Code patterns, style, error handling
3. `architecture/API_CONTRACTS.md` - API specs, data models, schemas

These documents contain all the details. This file is just a quick reference.

## System Architecture Quick Reference

### Deployable Units

**API Service** (`api/`)
- FastAPI server on port 8000
- Endpoints: POST /chat, POST /setup, GET /health
- LangGraph multi-agent (draft → critique)

**Background Job** (`background-job/`)
- Scheduled Python script (daily at midnight)
- Syncs Confluence pages incrementally

### Shared Components

**RAG Pipeline** (`shared/rag-pipeline/`)
- Importable Python module (NOT a microservice)
- Methods: retrieve(), process_page(), sync_space()

**Config** (`shared/config/`)
- Pydantic Settings - use get_settings(), never os.getenv()

## Key Architectural Principles

1. **Idempotency**: All operations safe to retry
2. **Environment Parity**: Dev (Ollama) matches Prod (OpenAI)
3. **Shared Code**: Import RAG pipeline, don't duplicate
4. **Immutable State**: LangGraph nodes return new dicts

## Data Flow Summary

**Question Answering**: User → API → RAG retrieve → Draft agent → Critic agent → Confidence check → (notify HR if low) → Response

**Initial Setup**: API → Fetch all pages → Check hash → Process changed pages → Store

**Daily Sync**: Midnight → Fetch pages → Check timestamps → Process outdated → Skip current

## Technology Stack

- Python 3.11+, UV workspace
- FastAPI, LangChain, LangGraph
- Qdrant (vectors), PostgreSQL (metadata)
- Ollama (dev), OpenAI (prod)

## Quick Task Checklist

Before completing any task:

- [ ] Read the 3 architecture documents
- [ ] Understand which service/component you're working on
- [ ] Check existing code for patterns
- [ ] Follow patterns from CODING_STANDARDS.md
- [ ] Match specs from API_CONTRACTS.md exactly
- [ ] Verify all code follows CODING_STANDARDS.md

## Core Patterns Reference

**For detailed patterns, see CODING_STANDARDS.md**

Key reminders only:
- Type hints everywhere
- Use get_settings() from hr_chatbot_config
- Check settings.llm.is_local for environment
- LangGraph: return new dicts, don't mutate state
- Use logger, never print()
- Structured error handling with HTTPException

## Configuration

Settings managed by `hr_chatbot_config` package:
- settings.environment (development/production)
- settings.database.connection_string
- settings.qdrant.url
- settings.llm.is_local
- settings.confluence.token

## Critical Reminders

1. **ALWAYS read architecture docs first** - they contain the source of truth
2. **This file is a quick reference** - when in doubt, check the detailed docs
3. **Don't duplicate patterns** - look at existing code first
4. **API contracts are strict** - match API_CONTRACTS.md exactly
5. **Environment awareness** - always check settings.llm.is_local

## Where to Find Details

- **System design & flows**: `architecture/ARCHITECTURE.md`
- **Code patterns & style**: `architecture/CODING_STANDARDS.md`
- **API specs & schemas**: `architecture/API_CONTRACTS.md`
- **This file**: Quick reference only - not the source of truth