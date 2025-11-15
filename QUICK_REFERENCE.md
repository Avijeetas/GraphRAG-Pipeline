# GraphRAG Pipeline - Quick Reference

## 🚀 Quick Start

```python
from graphrag_pipeline import GraphRAGPipeline

# 1. Check Ollama models (optional)
models = GraphRAGPipeline.list_ollama_models()
print(models)

# 2. Initialize (auto-checks and pulls models if enabled)
pipeline = GraphRAGPipeline(
    check_models=True,      # Check for required models
    auto_pull_models=True   # Auto-pull missing models
)

# 3. Process documents
result = pipeline.process_documents(file_path="your_doc.txt")

# 4. Query
answer = pipeline.full_retriever("Your question here")
```

---

## 📝 Common Operations

### Process Documents

```python
# Single file
result = pipeline.process_documents(file_path="doc.txt")

# Directory
result = pipeline.process_documents(directory="./docs", file_type="txt")

# Custom chunking
result = pipeline.process_documents(
    file_path="doc.txt",
    chunk_size=500,
    chunk_overlap=50
)
```

### Load Documents

```python
# Single file
docs = pipeline.load_documents(file_path="doc.txt")

# Directory
docs = pipeline.load_documents(directory="./docs", glob_pattern="*.txt")

# PDF files
docs = pipeline.load_documents(file_path="doc.pdf")
docs = pipeline.load_documents(directory="./pdfs", file_type="pdf")
```

### Split Documents

```python
documents = pipeline.split_documents(
    docs,
    chunk_size=250,
    chunk_overlap=24
)
```

### Extract Graph

```python
graph_docs = pipeline.extract_graph_structure(documents)
pipeline.store_graph(graph_docs)
```

### Create Indexes

```python
# Vector index
retriever = pipeline.create_vector_index(retrieval_kwargs={"k": 5})

# Full-text index
pipeline.create_fulltext_index()
```

### Retrieve Information

```python
# Graph only
graph_context = pipeline.graph_retriever("Who is X?")

# Graph + Vector (full)
full_context = pipeline.full_retriever("Who is X?")
```

### Create RAG Chain

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """Answer based on context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": pipeline.full_retriever, "question": RunnablePassthrough()}
    | prompt
    | pipeline.llm
    | StrOutputParser()
)

answer = chain.invoke("Your question")
```

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### Custom Initialization

```python
pipeline = GraphRAGPipeline(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password",
    llm_model="llama3.1",
    embedding_model="mistral",
    check_models=True,      # Check Ollama models
    auto_pull_models=True   # Auto-pull missing models
)
```

### Check and Pull Ollama Models

```python
# Check if Ollama is available
available = GraphRAGPipeline.check_ollama_available()
print(f"Ollama available: {available}")

# List all available models
models = GraphRAGPipeline.list_ollama_models()
for model in models:
    print(f"  - {model['name']}")

# Check for specific models
status = GraphRAGPipeline.ensure_ollama_models(
    required_models=["llama3.1", "mistral"],
    auto_pull=True,   # Auto-pull missing
    check_only=False  # Actually pull if missing
)

# Pull a specific model
GraphRAGPipeline.pull_ollama_model("llama3.1")
```

### Custom Vector Index

```python
retriever = pipeline.create_vector_index(
    search_type="hybrid",           # "hybrid", "vector", "keyword"
    node_label="Document",
    text_properties=["text"],
    embedding_property="embedding",
    retrieval_kwargs={"k": 5}       # Top k results
)
```

---

## 🔧 Customization

### Custom Entity Extraction

```python
from pydantic import BaseModel, Field

class CustomEntities(BaseModel):
    products: list[str] = Field(description="Product names")
    customers: list[str] = Field(description="Customer names")

pipeline.setup_entity_extraction()  # Use custom class in setup_entity_extraction
```

### Custom Graph Query

```python
def custom_retriever(question: str):
    entities = pipeline.entity_chain.invoke(question)
    result = ""
    
    for entity in entities.names:
        response = pipeline.graph.query(
            """
            MATCH (e:__Entity__ {id: $entity})
            OPTIONAL MATCH (e)-[r]->(related)
            RETURN e.id, type(r), related.id
            LIMIT 10
            """,
            {"entity": entity}
        )
        # Process response
    return result
```

---

## 📊 Pipeline Flow

```
1. Load Documents
   ↓
2. Split into Chunks
   ↓
3. Extract Entities & Relationships (LLM)
   ↓
4. Store in Neo4j Graph
   ↓
5. Create Vector Embeddings
   ↓
6. Create Indexes
   ↓
7. Ready for Querying
```

---

## 🎯 Retrieval Types

### Graph Retrieval
- Finds entities in question
- Traverses relationships
- Returns structured graph info

### Vector Retrieval
- Semantic similarity search
- Finds relevant document chunks
- Returns contextually similar content

### Hybrid Retrieval
- Combines graph + vector
- Most comprehensive
- Best for complex questions

---

## 📏 Recommended Parameters

### Chunk Sizes
- **Short documents** (emails, notes): 200-400 chars
- **Medium documents** (articles): 250-500 chars
- **Long documents** (books, papers): 500-1000 chars

### Overlap
- **10-20%** of chunk_size
- Example: chunk_size=500 → overlap=50-100

### Retrieval Limits
- **Vector**: k=3-5 (top results)
- **Graph**: LIMIT=20-50 (relationship chains)

---

## 🔍 Querying Tips

1. **Extract entities first**: Helps identify what to search
2. **Use graph for relationships**: "How are X and Y related?"
3. **Use vector for concepts**: "Explain what X means"
4. **Combine both**: "Who is X and what did they do?"

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection error | Check Neo4j Desktop is running |
| No embeddings | Verify Ollama model is available |
| Poor results | Adjust chunk sizes or retrieval limits |
| Slow processing | Reduce chunk_size or use GPU |

---

## 📚 File Structure

```
your_project/
├── .env                      # Environment variables
├── graphrag_pipeline.py      # Main pipeline module
├── example_usage.py          # Usage examples
├── PIPELINE_GUIDE.md         # Detailed guide
├── QUICK_REFERENCE.md        # This file
└── documents/                # Your documents
    ├── doc1.txt
    └── doc2.txt
```

---

## 🎓 Next Steps

1. **Process your documents**: `pipeline.process_documents()`
2. **Test retrieval**: `pipeline.full_retriever("test question")`
3. **Create RAG chain**: See example above
4. **Deploy**: Create API or web interface

---

**For detailed explanations, see `PIPELINE_GUIDE.md`**

