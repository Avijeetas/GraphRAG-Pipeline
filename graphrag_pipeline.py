"""
GraphRAG Pipeline - Reusable Module
A complete pipeline for building knowledge graphs from documents and performing hybrid RAG.
"""

import os
import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import argparse
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LangChain imports
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector

# Load environment variables
load_dotenv()


class GraphRAGPipeline:
    """
    Complete GraphRAG pipeline for document processing, knowledge graph creation,
    and hybrid retrieval (graph + vector).
    """
    
    @staticmethod
    def check_ollama_available() -> bool:
        """
        Check if Ollama is installed and running.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    @staticmethod
    def list_ollama_models() -> List[Dict[str, Any]]:
        """
        List all available Ollama models.
        
        Returns:
            List of dictionaries containing model information
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return []
            
            models = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        models.append({
                            "name": parts[0],
                            "id": parts[1] if len(parts) > 1 else None,
                            "size": parts[2] if len(parts) > 2 else None,
                            "modified": ' '.join(parts[3:]) if len(parts) > 3 else None
                        })
            return models
        except Exception as e:
            print(f"Error listing Ollama models: {e}")
            return []
    
    @staticmethod
    def pull_ollama_model(model_name: str) -> bool:
        """
        Pull a model from Ollama if not already available.
        
        Args:
            model_name: Name of the model to pull (e.g., "llama3.1")
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Pulling Ollama model: {model_name}...")
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                print(f"✓ Successfully pulled model: {model_name}")
                return True
            else:
                print(f"✗ Failed to pull model {model_name}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout while pulling model: {model_name}")
            return False
        except Exception as e:
            print(f"✗ Error pulling model {model_name}: {e}")
            return False
    
    @staticmethod
    def ensure_ollama_models(
        required_models: List[str],
        auto_pull: bool = True,
        check_only: bool = False
    ) -> Dict[str, bool]:
        """
        Ensure required Ollama models are available.
        
        Args:
            required_models: List of model names to check/pull
            auto_pull: If True, automatically pull missing models
            check_only: If True, only check availability without pulling
        
        Returns:
            Dictionary mapping model names to availability status
        """
        if not GraphRAGPipeline.check_ollama_available():
            print("⚠ Warning: Ollama is not available or not running")
            print("Please ensure Ollama is installed and running:")
            print("  1. Install Ollama from https://ollama.ai")
            print("  2. Start Ollama service")
            print("  3. Run: ollama list")
            return {model: False for model in required_models}
        
        available_models = GraphRAGPipeline.list_ollama_models()
        available_model_names = [model["name"] for model in available_models]
        
        status = {}
        missing_models = []
        
        for model in required_models:
            # Check if model exists (exact match or prefix match)
            is_available = any(
                available.startswith(model) or model == available
                for available in available_model_names
            )
            status[model] = is_available
            
            if not is_available:
                missing_models.append(model)
        
        if missing_models and not check_only:
            print(f"\n⚠ Missing models: {', '.join(missing_models)}")
            
            if auto_pull:
                print("\nAttempting to pull missing models...")
                for model in missing_models:
                    if GraphRAGPipeline.pull_ollama_model(model):
                        status[model] = True
                        # Update available models list after pulling
                        available_models = GraphRAGPipeline.list_ollama_models()
                        available_model_names = [m["name"] for m in available_models]
            else:
                print("\nTo pull missing models, run:")
                for model in missing_models:
                    print(f"  ollama pull {model}")
        
        return status
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        llm_model: str = "llama3.1",
        embedding_model: str = "mxbai-embed-large",
        check_models: bool = True,
        auto_pull_models: bool = True,
    ):
        """
        Initialize the GraphRAG pipeline.
        
        Args:
            neo4j_uri: Neo4j connection URI (defaults to env var NEO4J_URI)
            neo4j_username: Neo4j username (defaults to env var NEO4J_USERNAME)
            neo4j_password: Neo4j password (defaults to env var NEO4J_PASSWORD)
            llm_model: LLM model name for Ollama
            embedding_model: Embedding model name for Ollama
            check_models: If True, check and optionally pull required models
            auto_pull_models: If True, automatically pull missing models
        """
        # Store model names
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        # Check and ensure Ollama models are available
        if check_models:
            print("Checking Ollama models...")
            required_models = [llm_model, embedding_model]
            model_status = self.ensure_ollama_models(
                required_models,
                auto_pull=auto_pull_models,
                check_only=False
            )
            
            # Check if all models are available
            all_available = all(model_status.values())
            if not all_available:
                missing = [model for model, status in model_status.items() if not status]
                print(f"\n⚠ Warning: Some models are not available: {missing}")
                print("Pipeline may fail if models are not pulled.")
        
        # Initialize Neo4j connection
        self.graph = Neo4jGraph(
            url=neo4j_uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
            username=neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j"),
            password=neo4j_password or os.getenv("NEO4J_PASSWORD", "your_password")
        )
        
        # Initialize LLM and embeddings
        self.llm = ChatOllama(model=llm_model, temperature=0, format="json")
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        
        self.vector_retriever = None
        self.entity_chain = None
    
    def load_documents(
        self,
        file_path: Optional[str] = None,
        directory: Optional[str] = None,
        glob_pattern: str = "*.txt"
    ) -> List[Document]:
        """
        Load .txt documents from file or directory.
        
        Args:
            file_path: Path to a single .txt file
            directory: Path to directory containing .txt documents
            glob_pattern: Glob pattern for directory loading (default: "*.txt")
        
        Returns:
            List of Document objects
        
        Raises:
            FileNotFoundError: If file_path or directory doesn't exist
            ValueError: If neither file_path nor directory is provided
        """
        if file_path:
            # Check if file exists
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                logger.error("File not found: %s", file_path)
                logger.error("Absolute path: %s", file_path_obj.absolute())
                logger.error("Current working directory: %s", os.getcwd())
                print(f"\n❌ ERROR: File not found: {file_path}")
                print(f"   Absolute path: {file_path_obj.absolute()}")
                print(f"   Current directory: {os.getcwd()}")
                sys.exit(1)
            
            if not file_path_obj.is_file():
                logger.error("Path exists but is not a file: %s", file_path)
                print(f"\n❌ ERROR: Path exists but is not a file: {file_path}")
                sys.exit(1)
            
            # Verify it's a .txt file
            if not file_path_obj.suffix.lower() == '.txt':
                logger.warning("File %s is not a .txt file. Loading anyway...", file_path)
            
            logger.info("Loading .txt file: %s", file_path)
            try:
                loader = TextLoader(file_path=file_path)
                docs = loader.load()
                logger.info("Successfully loaded %d document(s) from %s", len(docs), file_path)
                return docs
            except (FileNotFoundError, IOError, ValueError) as e:
                logger.error("Error loading file %s: %s", file_path, str(e))
                print(f"\n❌ ERROR: Failed to load file {file_path}: {str(e)}")
                sys.exit(1)
        
        elif directory:
            # Check if directory exists
            directory_path = Path(directory)
            if not directory_path.exists():
                logger.error("Directory not found: %s", directory)
                logger.error("Absolute path: %s", directory_path.absolute())
                logger.error("Current working directory: %s", os.getcwd())
                print(f"\n❌ ERROR: Directory not found: {directory}")
                print(f"   Absolute path: {directory_path.absolute()}")
                print(f"   Current directory: {os.getcwd()}")
                sys.exit(1)
            
            if not directory_path.is_dir():
                logger.error("Path exists but is not a directory: %s", directory)
                print(f"\n❌ ERROR: Path exists but is not a directory: {directory}")
                sys.exit(1)
            
            logger.info("Loading .txt documents from directory: %s (pattern: %s)", directory, glob_pattern)
            try:
                loader = DirectoryLoader(directory, glob=glob_pattern)
                docs = loader.load()
                logger.info("Successfully loaded %d document(s) from %s", len(docs), directory)
                return docs
            except (FileNotFoundError, IOError, ValueError) as e:
                logger.error("Error loading documents from directory %s: %s", directory, str(e))
                print(f"\n❌ ERROR: Failed to load documents from {directory}: {str(e)}")
                sys.exit(1)
        
        else:
            error_msg = "Either file_path or directory must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def split_documents(
        self,
        documents: List[Document],
        chunk_size: int = 250,
        chunk_overlap: int = 24
    ) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        
        Returns:
            List of chunked Document objects
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents=documents)
    
    def extract_graph_structure(self, documents: List[Document]) -> List[Any]:
        """
        Extract entities and relationships from documents using LLM.
        
        Args:
            documents: List of Document objects
        
        Returns:
            List of GraphDocument objects containing nodes and relationships
        """
        graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
        return graph_documents
    
    def store_graph(self, graph_documents: List[Any], base_entity_label: bool = True, include_source: bool = True) -> None:
        """
        Store extracted graph structure in Neo4j.
        
        Args:
            graph_documents: List of GraphDocument objects
            base_entity_label: Whether to add base __Entity__ label
            include_source: Whether to link nodes to source documents
        """
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=base_entity_label,
            include_source=include_source
        )
    
    def create_vector_index(
        self,
        search_type: str = "hybrid",
        node_label: str = "Document",
        text_properties: List[str] = None,
        embedding_property: str = "embedding",
        retrieval_kwargs: Dict = {"k": 3}
    ) -> Any:
        """
        Create vector embeddings and retriever for hybrid search.
        
        Args:
            search_type: Type of search ("hybrid", "vector", "keyword")
            node_label: Label of nodes to embed
            text_properties: List of properties containing text
            embedding_property: Property name to store embeddings
            retrieval_kwargs: Additional kwargs for retriever (e.g., {"k": 5})
        
        Returns:
            Vector retriever object
        """
        if text_properties is None:
            text_properties = ["text"]
        
        vector_index = Neo4jVector.from_existing_graph(
            self.embeddings,
            search_type=search_type,
            node_label=node_label,
            text_node_properties=text_properties,
            embedding_node_property=embedding_property,
        )
        
        if retrieval_kwargs:
            self.vector_retriever = vector_index.as_retriever(search_kwargs=retrieval_kwargs)
        else:
            self.vector_retriever = vector_index.as_retriever()
        
        return self.vector_retriever
    def create_fulltext_index(self) -> None:
        """
        Create full-text search index on entity IDs for fast entity lookup.
        """
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"),
                  os.getenv("NEO4J_PASSWORD", "your_password"))
        )

        def index_exists_tx(tx):
            try:
                # Try Neo4j 5.x syntax first
                res = tx.run("SHOW INDEXES YIELD name, type WHERE type = 'FULLTEXT' AND name = 'fulltext_entity_id' RETURN name")
                return any(record["name"] == "fulltext_entity_id" for record in res)
            except Exception:
                # Fall back to Neo4j 4.x syntax
                try:
                    res = tx.run(
                        """
                        CALL db.indexes() YIELD name, type
                        WHERE type = 'FULLTEXT' AND name = 'fulltext_entity_id'
                        RETURN name
                        """
                    )
                    return any(record["name"] == "fulltext_entity_id" for record in res)
                except Exception:
                    # If both fail, assume index doesn't exist
                    return False

        def create_fulltext_index_tx(tx):
            # Drop index if it exists (in case of partial creation)
            try:
                tx.run("DROP INDEX fulltext_entity_id IF EXISTS")
            except Exception:
                pass
            
            # Create the index (syntax compatible with Neo4j 4.x and 5.x)
            query = '''
            CREATE FULLTEXT INDEX fulltext_entity_id
            FOR (n:__Entity__)
            ON EACH [n.id]
            '''
            tx.run(query)

        try:
            with driver.session() as session:
                # Check if index already exists first
                exists = session.execute_read(index_exists_tx)
                if exists:
                    print("✓ Fulltext index already exists.")
                    return
                
                # Create index
                print("Creating fulltext index...")
                session.execute_write(create_fulltext_index_tx)
                
                # Wait a moment for index to be created
                import time
                time.sleep(2)
                
                # Confirm creation
                exists = session.execute_read(index_exists_tx)
                if exists:
                    print("✓ Fulltext index created successfully.")
                else:
                    print("⚠ Warning: Could not confirm index creation. It may still be building.")
                    print("  The index should be available shortly.")
        except Exception as e:
            print(f"⚠ Warning: Error during index creation: {e}")
            print("  Attempting to continue anyway. If queries fail, you may need to create the index manually:")
            print("  CREATE FULLTEXT INDEX fulltext_entity_id FOR (n:__Entity__) ON EACH [n.id]")
        finally:
            driver.close()
    def setup_entity_extraction(self) -> None:
        """
        Setup entity extraction chain for question processing.
        """
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
        
        self.entity_chain = self.llm.with_structured_output(Entities)
    
    def graph_retriever(self, question: str, limit: int = 50) -> str:
        """
        Retrieve graph context for a question by finding entities and their relationships.
        
        Args:
            question: User question
            limit: Maximum number of results to return
        
        Returns:
            Formatted string containing graph relationships
        """
        if self.entity_chain is None:
            self.setup_entity_extraction()
        
        result = ""
        entities = self.entity_chain.invoke(question)
        
        for entity in entities.names:
            response = self.graph.query(
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
                RETURN output LIMIT $limit
                """,
                {"query": entity, "limit": limit},
            )
            result += "\n".join([el['output'] for el in response])
        
        return result

    def full_retriever(self, question: str) -> str:
        """
        Combine graph and vector retrieval for comprehensive context.
        
        Args:
            question: User question
        
        Returns:
            Combined context from graph and vector retrieval, with safety checks
        """
        # Graph context
        graph_data = self.graph_retriever(question) if self.graph else "⚠ Graph not initialized."
        vector_data = []
        # Vector context
        if self.vector_retriever:
            try:
                vector_docs = self.vector_retriever.invoke(question)
                vector_data = [doc.page_content.replace("text:","") for doc in vector_docs] if vector_docs else ["⚠ No vector data found."]
            except Exception as e:
                vector_data = [f"⚠ Error retrieving vector data: {e}"]
        else:
            vector_data = ["⚠ Vector retriever not initialized."]
        
        if len(vector_data) == 0:
            print("no document is found")
        return {
            "Relationship": graph_data,
            "Document": "\n".join(vector_data)
        }

    
    def process_document(
        self,
        file_path: Optional[str] = None,
        directory: Optional[str] = None,
        chunk_size: int = 150,
        chunk_overlap: int = 24,
        create_vector_index: bool = True,
        create_fulltext_index: bool = True
    ) -> Dict[str, Any]:
        """
        Complete pipeline: load → split → extract → store → index.
        
        Args:
            file_path: Path to a single document
            directory: Path to directory containing documents
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            create_vector_index: Whether to create vector embeddings
            create_fulltext_index: Whether to create full-text index
        
        Returns:
            Dictionary containing processed documents and retrievers
        """
        # Step 1: Load documents
        # Note: load_documents will exit with error log if file_path/directory doesn't exist
        logger.info("Loading documents...")
        print("Loading documents...")
        docs = self.load_documents(file_path=file_path, directory=directory)
        logger.info("Loaded %d documents", len(docs))
        print(f"Loaded {len(docs)} documents")
        
        # Step 2: Split documents
        print("Splitting documents...")
        documents = self.split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"Splitted into {len(documents)} chunks")
        
        # Step 3: Extract graph structure
        print("Extracting entities and relationships...")
        graph_documents = self.extract_graph_structure(documents)
        print(f"Extracted {len(graph_documents)} graph documents")
        
        # Step 4: Store graph
        print("Storing graph in Neo4j...")
        self.store_graph(graph_documents)
        print("Graph stored successfully")
        
        # Step 5: Create vector index
        if create_vector_index:
            print("Creating vector embeddings...")
            self.create_vector_index()
            print("Vector index created")
        
        # Step 6: Create full-text index
        if create_fulltext_index:
            print("Creating full-text index...")
            self.create_fulltext_index()
        
        # Step 7: Setup entity extraction
        self.setup_entity_extraction()
        
        return {
            "documents": documents,
            "graph_documents": graph_documents,
            "vector_retriever": self.vector_retriever
        }

