# RAG Research Assistant Vera

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-purple.svg)
![LangGraph](https://img.shields.io/badge/langgraph-latest-9cf.svg)
![MistralAI](https://img.shields.io/badge/mistralai-supported-lightgrey.svg)
![arXiv API](https://img.shields.io/badge/arxiv--api-integrated-yellowgreen.svg)
![ChromaDB](https://img.shields.io/badge/chromadb-0.4+-orange.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-red.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="images/app.jpg" alt="App Image">
</p>

**Vera** is a local Retrieval-Augmented Generation (RAG) agent designed for researchers and students to query recent Machine Learning and AI publications from sources like arXiv. It combines modern language models with vector-based document retrieval to generate context-aware responses.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Example Usage](#example-usage)
- [Project Structure](#project-structure)
- [Status](#status)

## Overview

Vera is an agentic RAG system that:

- **Retrieves** relevant documents from a vector database based on user queries via similarity search

- **Augments** the query with retrieved context

- **Generates** responses using a language model with the retrieved context

The system leverages LangChain, LangGraph, and ChromaDB for modular RAG pipelines. Currently designed for **local deployment**, but can be adapted for production with API and front-end support.

## Architecture

A routing graph node handles conversational flow. The agent uses two key tools:

- `retrieve`: fetches research papers from the vector store

- `lookup`: queries metadata or direct paper content

<p align="center">
  <img src="images/architecture.svg" alt="Architecture Image">
</p>

The graph conditionally selects tools before generating a final response using Mistral.

## Installation

### Prerequisites

- Python 3.10+

- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip` package manager

- `venv` or Docker (optional but recommended for isolated setup)

### Setup

1. **Clone the repository**

    ```bash
    git clone https://github.com/hml230/rag-agent-project.git
    cd vera-rag
    ```

2. **Setup virtual environment** or **Docker**

    First, setup your environment variables

    ```bash
    cp .env.example .env # edit .env with your configuration
    ```

    Next, run

    ```bash
    python -m venv .venv
    source venv/bin/activate
    uv pip install -r requirements.txt # use uv for quicker install
    ```

    or

    ```bash
    docker build -t vera .
    ```

    **Note:** Initial Docker image may exceed 6GB due to Torch/HuggingFace deps. Some NVIDIA packages may be required even if GPU is unused.

    Finally, create a `config.yaml` file to store the paths of your storages, the storages are currently embedded and not external.

    ```yaml
    db_paths:
      chroma: data/chroma_langchain_db/
      sqlite: data/
    ```

3. **Running the agent**

    If all requirements are installed, running `app.py` will initialise storages and the agent's chat loop at your specified paths.

      ```bash
      python3 src/app.py
      ```

    Alternatively, the built container can be started using

      ```bash
      docker compose up --build
      ```

    The app is now running on port 5000, you can access the front-end in the browser.

## Example Usage

Simply input your query as display below, the agent will respond in lower chatbox.

<p align="center">
  <img src="images/usage.jpg" alt="Usage Image 1" width=1080>
</p>

Another exmaple,

<p align="center">
  <img src="images/usage01.jpg" alt="Usage Image 2" width=1080>
</p>

Logs and relevant messages will be displayed in the terminal during the conversation.

To terminate the agent, input `Ctrl+C` in your terminal session.

## Addtional Configuration

Extra configurations can be added to the `config.yaml` file to manage settings:

```yaml
# Vector Database Configuration
vector_db:
  type: "chroma"
  path: "./data/vectordb"
  collection_name: "documents"

# Embedding Model Configuration
embeddings:
  model: "mistral-embed"

# Language Model Configuration
llm:
  provider: "mistralai" # or "huggingface", "anthropic"
  model: "mistral-small-2503"

# Retrieval Configuration
retrieval:
  top_k: 5
  similarity_threshold: 0.7
  chunk_size: 1000
  chunk_overlap: 200
```

## Project Structure

```text
vera
├── Dockerfile
├── README.md
├── config.yaml
├── docker-compose.yaml
├── images
│   ├── Architecture.drawio.svg
│   ├── app.jpg
│   ├── usage.jpg
│   └── usage01.jpg
├── pyproject.toml
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── app.py
│   ├── main.py
│   ├── query.py
│   ├── storage.py
│   ├── templates
│   │   └── index.html
│   └── vector_store.py
└── uv.lock
```

## Status

### Completed

- Core RAG pipeline implementation

- Local storage and retrieval

- CLI and browser chat interface

- Docker support

### Future Development

- Deployable Flask API

- Chat history and caching with LangGraph

- Test suite

- System evaluation metrics, agent monitoring components (drift, hallucinations, etc.)

- More advanced retrieval strategies

## Acknowledgments

- Built with [LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/) and [LangChain](https://python.langchain.com/docs/introduction/) RAG documentations.

- Uses [ChromaDB](https://github.com/chroma-core/chroma) for vector storage

- Powered by [Mistral 24B Small](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) and `mistral-embed` models

- Thank you to arXiv for use of its open access interoperability

---

**Note**: The Flask API is currently in development and not ready for production use. For deployment, an API Gateway and additional development are needed.
