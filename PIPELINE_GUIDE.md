# GraphRAG Pipeline - Step-by-Step Guide

## 📋 Table of Contents
1. [Pipeline Overview](#pipeline-overview)
2. [Architecture Explained](#architecture-explained)
3. [Setup Instructions](#setup-instructions)
4. [Pipeline Steps Breakdown](#pipeline-steps-breakdown)
5. [Adapting for Your Project](#adapting-for-your-project)
6. [Using as a Reusable Pipeline](#using-as-a-reusable-pipeline)

---

## 🎯 Pipeline Overview

This GraphRAG pipeline combines **Graph Knowledge** (Neo4j) with **Vector Search** (embeddings) to create a powerful Retrieval Augmented Generation (RAG) system. It extracts entities and relationships from documents, stores them in a knowledge graph, and uses both graph traversal and semantic search to retrieve relevant context for answering questions.

### Key Components:
- **Document Loading**: Load your text documents
- **Text Splitting**: Chunk documents for processing
- **Graph Extraction**: Extract entities and relationships using LLM
- **Graph Storage**: Store in Neo4j knowledge graph
- **Vector Embeddings**: Create semantic embeddings for hybrid search
- **Hybrid Retrieval**: Combine graph-based and vector-based retrieval
- **Answer Generation**: Generate answers using retrieved context

---

## 🏗️ Architecture Explained

```
┌─────────────┐
│  Documents  │
│   (TXT)     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Text Splitting  │  ← Chunks documents into manageable pieces
└──────┬──────────┘
       │
       ▼
┌──────────────────────┐
│  LLM Graph Extractor │  ← Extracts entities & relationships
└──────┬───────────────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│   Neo4j     │◄─────┤ Vector Store │  ← Stores graph + embeddings
│   Graph     │      │  (Embedded)  │
└──────┬──────┘      └──────┬───────┘
       │                     │
       │                     │
       ▼                     ▼
┌──────────────────────────────────┐
│     Hybrid Retriever             │  ← Combines graph + vector search
│  (Graph Traversal + Semantic)    │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────┐
│   Question → Context     │  ← Retrieves relevant information
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│   LLM Answer Generation  │  ← Generates final answer
└──────────────────────────┘
```

---

## ⚙️ Setup Instructions

### Step 1: Environment Setup

1. **Create `.env` file** in project root: I am using neo4j desktop version
```bash
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Start Neo4j Desktop**:
   - Open Neo4j Desktop
   - Create/start your database
   - Enable APOC plugin (in database settings)

4. **Start Ollama** and ensure models are available:
```bash
# Check if Ollama is running
ollama list

# Pull required models if not available
ollama pull llama3.1
ollama pull mxbai-embed-large
```

### Step 2: Prepare Your Documents

Place your documents in a folder or single file:
- `.txt` files (plain text)
- Other supported formats

---

## 📚 Pipeline Steps Breakdown

### **Step 1: Initialize Connections**

```python
from dotenv import load_dotenv
import os
from langchain_neo4j import Neo4jGraph

# Load environment variables
load_dotenv()

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "your_password")
)
```

**What it does**: Establishes connection to your Neo4j database where the knowledge graph will be stored.

---

### **Step 2: Load and Split Documents**

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load document
loader = TextLoader(file_path="your_document.txt")
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,      # Characters per chunk
    chunk_overlap=24     # Overlap between chunks (for context continuity)
)
documents = text_splitter.split_documents(documents=docs)
```

**Parameters to adjust**:
- `chunk_size`: Size of each text chunk (250-1000 chars recommended)
- `chunk_overlap`: Overlap between chunks (10-20% of chunk_size)

**What it does**: 
- Loads your documents
- Splits them into smaller chunks that can be processed by LLM
- Maintains context through overlapping chunks

---

### **Step 3: Extract Graph Structure (Entities & Relationships)**

```python
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Initialize LLM for graph extraction
llm = ChatOllama(model="llama3.1", temperature=0, format="json")

# Create graph transformer
llm_transformer = LLMGraphTransformer(llm=llm)

# Extract entities and relationships
graph_documents = llm_transformer.convert_to_graph_documents(documents)
```

**What it does**:
- Uses LLM to analyze each document chunk
- Identifies entities (people, places, organizations, concepts)
- Identifies relationships between entities
- Returns structured graph data (nodes and edges)

**Output example**:
```
GraphDocument(
    nodes=[
        Node(id="Nonna Lucia", type='Person', properties={}),
        Node(id="Amico", type='Person', properties={})
    ],
    relationships=[
        Relationship(source="Nonna Lucia", target="Amico", type="TAUGHT")
    ]
)
```

---

### **Step 4: Store Graph in Neo4j**

```python
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,    # Adds base __Entity__ label to all entities
    include_source=True      # Links nodes back to source documents
)
```

**What it does**:
- Stores all extracted entities as nodes in Neo4j
- Stores relationships as edges
- Links entities to source documents for traceability
- Creates indexes for fast retrieval

---

### **Step 5: Create Vector Embeddings**

```python
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector

# Initialize embeddings
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"  # Embedding model
)

# Create vector index from existing graph
vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",              # hybrid = vector + keyword search
    node_label="Document",             # Node type to embed
    text_node_properties=["text"],     # Property containing text
    embedding_node_property="embedding" # Property to store embeddings
)

# Create retriever
vector_retriever = vector_index.as_retriever()
```

**What it does**:
- Creates semantic embeddings for all document nodes
- Stores embeddings in Neo4j
- Enables semantic similarity search
- Hybrid search combines vector similarity + keyword matching

---

### **Step 6: Create Full-Text Index for Graph Entities**

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
    auth=(os.getenv("NEO4J_USERNAME", "neo4j"),
          os.getenv("NEO4J_PASSWORD", "your_password"))
)

def create_fulltext_index(tx):
    query = '''
    CREATE FULLTEXT INDEX `fulltext_entity_id` 
    FOR (n:__Entity__) 
    ON EACH [n.id];
    '''
    tx.run(query)

def create_index():
    with driver.session() as session:
        session.execute_write(create_fulltext_index)
        print("Fulltext index created successfully.")

try:
    create_index()
except:
    pass  # Index might already exist

driver.close()
```

**What it does**:
- Creates a full-text search index on entity IDs
- Enables fast entity lookup by name
- Used for graph-based retrieval

---

### **Step 7: Create Entity Extraction Chain**

```python
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

class Entities(BaseModel):
    """Identifying information about entities."""
    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting organization and person entities from the text."),
    ("human", "Use the given format to extract information from the following input: {question}"),
])

# Create entity extraction chain
entity_chain = llm.with_structured_output(Entities)
```

**What it does**:
- Extracts entity names from questions
- Uses structured output to ensure consistent format
- Helps identify which entities to query in the graph

---

### **Step 8: Create Graph Retriever**

```python
def graph_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned in the question.
    """
    result = ""
    entities = entity_chain.invoke(question)
    
    for entity in entities.names:
        response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result
```

**What it does**:
- Extracts entities from the question
- Finds matching entities in Neo4j using full-text search
- Retrieves all connected entities (neighbors) via relationships
- Returns structured graph information as text

**Query Explanation**:
- `db.index.fulltext.queryNodes`: Searches entities by name
- `MATCH (node)-[r:!MENTIONS]->(neighbor)`: Gets outgoing relationships
- `MATCH (node)<-[r:!MENTIONS]-(neighbor)`: Gets incoming relationships
- Returns relationship chains like "Entity1 - RELATIONSHIP_TYPE -> Entity2"

---

### **Step 9: Create Full Retriever (Graph + Vector)**

```python
def full_retriever(question: str):
    # Get graph-based context
    graph_data = graph_retriever(question)
    
    # Get vector-based context
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    
    # Combine both
    final_data = f"""Graph data:
{graph_data}
vector data:
{"#Document ".join(vector_data)}
    """
    return final_data
```

**What it does**:
- Combines graph traversal results with semantic search results
- Graph data provides structured relationship information
- Vector data provides semantically similar document chunks
- Together, they provide comprehensive context

---

### **Step 10: Create RAG Chain**

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Create the chain
chain = (
    {
        "context": full_retriever,           # Retrieves context
        "question": RunnablePassthrough(),   # Passes question through
    }
    | prompt                                 # Formats prompt
    | llm                                    # Generates answer
    | StrOutputParser()                      # Parses output
)

# Use the chain
answer = chain.invoke("Who is Nonna Lucia? Did she teach anyone?")
```

**What it does**:
- Combines retrieval with generation
- Retrieves relevant context (graph + vector)
- Formats prompt with context and question
- Generates answer using LLM
- Returns final answer

---

## 🔧 Adapting for Your Project

### 1. **Change Document Source**

```python
# For multiple files
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("./documents", glob="*.txt")
docs = loader.load()
```


### 2. **Adjust Text Splitting Parameters**

```python
# For longer documents (research papers, books)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# For shorter documents (emails, notes)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

### 3. **Customize Entity Extraction**

Modify the `Entities` class to extract domain-specific entities:

```python
class CustomEntities(BaseModel):
    products: list[str] = Field(description="Product names")
    customers: list[str] = Field(description="Customer names")
    locations: list[str] = Field(description="Location names")
    dates: list[str] = Field(description="Important dates")
```

### 4. **Modify Graph Retrieval Query**

Customize the Cypher query to match your domain:

```python
def custom_graph_retriever(question: str) -> str:
    entities = entity_chain.invoke(question)
    result = ""
    
    for entity in entities.names:
        # Custom query for your domain
        response = graph.query(
            """
            MATCH (e:__Entity__ {id: $entity})
            OPTIONAL MATCH (e)-[r]->(related)
            RETURN e.id as entity, type(r) as relation, related.id as related_entity
            LIMIT 20
            """,
            {"entity": entity},
        )
        # Format results as needed
        ...
    
    return result
```

### 5. **Adjust Retrieval Parameters**

```python
# Vector retriever with more results
vector_retriever = vector_index.as_retriever(
    search_kwargs={"k": 5}  # Retrieve top 5 most similar chunks
)

# With score threshold
vector_retriever = vector_index.as_retriever(
    search_kwargs={"k": 5, "score_threshold": 0.7}  # Only if similarity > 0.7
)
```

---

## 🔄 Using as a Reusable Pipeline

### Option 1: Create a Python Module

Create `graphrag_pipeline.py`:

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector

load_dotenv()

class GraphRAGPipeline:
    def __init__(self, neo4j_uri=None, neo4j_username=None, neo4j_password=None):
        """Initialize the pipeline with Neo4j connection."""
        self.graph = Neo4jGraph(
            url=neo4j_uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
            username=neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j"),
            password=neo4j_password or os.getenv("NEO4J_PASSWORD", "your_password")
        )
        
        self.llm = ChatOllama(model="llama3.1", temperature=0, format="json")
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
    
    def load_and_split_documents(self, file_path, chunk_size=250, chunk_overlap=24):
        """Load and split documents."""
        loader = TextLoader(file_path=file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents=docs)
    
    def build_knowledge_graph(self, documents):
        """Extract and store graph structure."""
        # Extract graph
        graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
        
        # Store in Neo4j
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        
        return graph_documents
    
    def create_vector_index(self):
        """Create vector embeddings for hybrid search."""
        vector_index = Neo4jVector.from_existing_graph(
            self.embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        return vector_index.as_retriever()
    
    def process_documents(self, file_path, chunk_size=250, chunk_overlap=24):
        """Complete pipeline: load → extract → store."""
        # Load and split
        documents = self.load_and_split_documents(file_path, chunk_size, chunk_overlap)
        
        # Build graph
        graph_documents = self.build_knowledge_graph(documents)
        
        # Create vector index
        vector_retriever = self.create_vector_index()
        
        return {
            "documents": documents,
            "graph_documents": graph_documents,
            "vector_retriever": vector_retriever
        }

# Usage
if __name__ == "__main__":
    pipeline = GraphRAGPipeline()
    result = pipeline.process_documents("your_document.txt")
```

### Option 2: Create a Configuration File

Create `config.yaml`:

```yaml
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "your_password"

llm:
  model: "llama3.1"
  temperature: 0

embeddings:
  model: "mxbai-embed-large"

text_splitter:
  chunk_size: 250
  chunk_overlap: 24

retrieval:
  vector_k: 5
  graph_limit: 50
```

### Option 3: Create a CLI Tool

Create `graphrag_cli.py`:

```python
import argparse
from graphrag_pipeline import GraphRAGPipeline

def main():
    parser = argparse.ArgumentParser(description="GraphRAG Pipeline CLI")
    parser.add_argument("--file", required=True, help="Document file to process")
    parser.add_argument("--chunk-size", type=int, default=250, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=24, help="Chunk overlap")
    
    args = parser.parse_args()
    
    pipeline = GraphRAGPipeline()
    result = pipeline.process_documents(
        args.file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    print(f"Processed {len(result['documents'])} documents")
    print(f"Extracted {len(result['graph_documents'])} graph documents")

if __name__ == "__main__":
    main()
```

Usage:
```bash
python graphrag_cli.py --file my_document.txt --chunk-size 500
```

---

## 📊 Workflow Summary

```
1. Setup
   ├── Install dependencies
   ├── Configure .env
   └── Start Neo4j & Ollama

2. Initialize
   ├── Connect to Neo4j
   ├── Load LLM & Embeddings
   └── Create transformers

3. Document Processing
   ├── Load documents
   ├── Split into chunks
   └── Extract graph structure

4. Storage
   ├── Store graph in Neo4j
   ├── Create embeddings
   └── Create indexes

5. Retrieval Setup
   ├── Create vector retriever
   ├── Create graph retriever
   └── Combine retrievers

6. Query
   ├── Extract entities from question
   ├── Retrieve graph context
   ├── Retrieve vector context
   └── Generate answer

7. Usage
   └── Ask questions via chain.invoke()
```

---

## 🎓 Best Practices

1. **Chunk Size**: 
   - Small chunks (200-500): Better for specific facts
   - Large chunks (500-1000): Better for context retention

2. **Chunk Overlap**:
   - 10-20% of chunk_size for better context continuity

3. **Entity Extraction**:
   - Customize entity types for your domain
   - Fine-tune prompts for better extraction

4. **Graph Queries**:
   - Optimize Cypher queries for your use case
   - Add relationship filters if needed

5. **Retrieval Balance**:
   - Adjust `k` parameter for vector search
   - Adjust `LIMIT` in graph queries
   - Balance graph vs vector results

6. **Monitoring**:
   - Check Neo4j browser for graph structure
   - Monitor retrieval quality
   - Adjust parameters based on results

---

## 🚀 Quick Start Template

```python
from graphrag_pipeline import GraphRAGPipeline

# Initialize
pipeline = GraphRAGPipeline()

# Process your documents
result = pipeline.process_documents("your_document.txt")

# Get retrievers
vector_retriever = result["vector_retriever"]

# Create query chain (use full_retriever from notebook)
# ... (as shown in Step 10)

# Query
answer = chain.invoke("Your question here")
print(answer)
```

---

## 📝 Next Steps

1. **Test with your documents**: Start with a small sample
2. **Tune parameters**: Adjust chunk sizes, retrieval limits
3. **Customize extraction**: Modify entity types for your domain
4. **Optimize queries**: Fine-tune Cypher queries for better results
5. **Scale up**: Process larger document sets
6. **Deploy**: Create API endpoints or web interface

---

## 🐛 Troubleshooting

**Issue**: Neo4j connection fails
- Check Neo4j Desktop is running
- Verify credentials in `.env`
- Check firewall settings

**Issue**: LLM extraction fails
- Verify Ollama is running: `ollama list`
- Check model is available: `ollama pull llama3.1`

**Issue**: Poor retrieval results
- Adjust chunk sizes
- Increase retrieval limits
- Fine-tune entity extraction prompts

**Issue**: Slow processing
- Reduce chunk_size for faster processing
- Use GPU-accelerated Ollama if available
- Optimize Neo4j indexes

---

**Happy Building! 🎉**

