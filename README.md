# GraphRAG with Llama 3.1

A project demonstrating Graph RAG (Retrieval Augmented Generation) using LangChain, Neo4j, and Llama 3.1.

## Setup

### 1. Neo4j Desktop Setup

Since you're using Neo4j Desktop, you don't need Docker Compose. Follow these steps:

1. **Start Neo4j Desktop** and create/start your database
2. **Get Connection Details:**
   - Open Neo4j Desktop
   - Select your database project
   - Click on "Details" or "Manage" to see connection details
   - Note the neo4j URI (typically `bolt://localhost:7687`)
   - Note your username (typically `neo4j`) and password

3. **Create `.env` file** in the project root with:
   ```
   NEO4J_URI=neo4j://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_actual_password
   ```

4. **Enable APOC Plugin** (if not already enabled):
   - In Neo4j Desktop, go to your database settings
   - Enable the APOC plugin (should be available in plugins)

### 2. Python Dependencies

Install all required packages by running the first cell in the notebook, or manually:
```bash
pip install langchain langchain-community langchain-ollama langchain-experimental langchain-text-splitters neo4j tiktoken yfiles_jupyter_graphs python-dotenv json-repair langchain-openai langchain_core
```

### 3. Ollama Setup

Make sure Ollama is running and you have the required models:
- `llama3.1` (for LLM functions)
- `mxbai-embed-large` (for embeddings)

## Usage

1. Ensure Neo4j Desktop database is running
2. Update the `.env` file with your Neo4j credentials
3. Run the notebook cells in order

## Note

The notebook will automatically use your `.env` file for Neo4j connection. If no `.env` file is found, it will default to `bolt://localhost:7687` with username `neo4j` and password `your_password` (you'll need to change this).
