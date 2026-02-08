# Architecture

## RAG Pipeline

```mermaid
flowchart LR
    subgraph Ingest [Ingestion]
        PDF[PDF/TXT] --> Load[Load]
        Load --> Chunk[Chunk]
        Chunk --> Embed[Embed]
        Embed --> Chroma[Chroma DB]
    end

    subgraph Query [Query Flow]
        Q[Question] --> Retrieve[Retrieve top-k]
        Retrieve --> Chroma
        Retrieve --> LLM[LLM]
        LLM --> Answer[Answer + Sources]
        Answer --> Gate[Human Review Gate]
    end

    subgraph Extract [Extraction Flow]
        Retrieve2[Retrieve] --> Chroma
        Retrieve2 --> ExtractLLM[Extraction LLM]
        ExtractLLM --> Validate[Validate Schema]
        Validate --> ReviewGate[Review Gate]
    end
```

## Multi-Agent Flow (LangGraph)

```mermaid
flowchart LR
    Extraction[Extraction Agent] --> Validation[Validation Agent]
    Validation --> Summary[Summary Agent]
    Summary --> Output[Output]
```

## System Overview

```mermaid
flowchart TB
    subgraph Client [Client]
        Streamlit[Streamlit UI]
        API_Client[API Client / curl]
    end

    subgraph Backend [Backend]
        FastAPI[FastAPI API]
        API_Client --> FastAPI
        Streamlit --> FastAPI
    end

    subgraph Core [Core]
        Ingest[Ingest]
        QA[Q&A]
        Extract[Extract]
        Agents[LangGraph Agents]
    end

    FastAPI --> Ingest
    FastAPI --> QA
    FastAPI --> Extract
    FastAPI --> Agents

    Ingest --> Chroma[(Chroma)]
    QA --> Chroma
    Extract --> Chroma
    Agents --> Chroma

    Ingest --> LLM[OpenAI / Azure OpenAI]
    QA --> LLM
    Extract --> LLM
    Agents --> LLM
```

## Tech Stack

| Layer | Component |
|-------|-----------|
| UI | Streamlit |
| API | FastAPI (async) |
| Orchestration | LangChain, LangGraph |
| Vector DB | Chroma |
| LLM | OpenAI / Azure OpenAI |
| Tracking | MLFlow |
