"""
Example usage of the GraphRAG Pipeline
This script demonstrates how to use the pipeline for your own documents.
"""

from graphrag_pipeline import GraphRAGPipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def check_ollama_models_example():
    """Example of checking and managing Ollama models."""
    print("=" * 60)
    print("Checking Ollama Models")
    print("=" * 60)
    
    # Check if Ollama is available
    if GraphRAGPipeline.check_ollama_available():
        print("✓ Ollama is available\n")
        
        # List all available models
        print("Current Ollama models:")
        models = GraphRAGPipeline.list_ollama_models()
        if models:
            for model in models:
                print(f"  - {model['name']} (ID: {model.get('id', 'N/A')})")
        else:
            print("  No models found")
        
        print("\n" + "-" * 60)
        
        # Check for required models
        required = ["llama3.1", "mistral"]
        print(f"\nChecking required models: {required}")
        status = GraphRAGPipeline.ensure_ollama_models(
            required_models=required,
            auto_pull=False,  # Set to True to auto-pull
            check_only=True   # Just check, don't pull
        )
        
        for model, available in status.items():
            status_icon = "✓" if available else "✗"
            print(f"  {status_icon} {model}: {'Available' if available else 'Missing'}")
    else:
        print("✗ Ollama is not available or not running")
        print("Please install Ollama from https://ollama.ai")


def main():
    """Main example workflow."""
    
    # Step 0: Check Ollama models (optional)
    print("=" * 60)
    print("Step 0: Checking Ollama Models (Optional)")
    print("=" * 60)
    check_ollama_models_example()
    
    # Step 1: Initialize the pipeline (will auto-check and pull models if enabled)
    print("\n" + "=" * 60)
    print("Step 1: Initializing GraphRAG Pipeline")
    print("=" * 60)
    print("Note: Pipeline will automatically check and pull models if needed")
    pipeline = GraphRAGPipeline(
        check_models=True,      # Check for required models
        auto_pull_models=True   # Automatically pull missing models
    )
    
    # Step 2: Process your documents
    print("\n" + "=" * 60)
    print("Step 2: Processing Documents")
    print("=" * 60)
    
    # Option A: Process a single file
    result = pipeline.process_documents(
        file_path="dummytext.txt",
        chunk_size=250,
        chunk_overlap=24
    )
    
    # Option B: Process multiple files in a directory
    # result = pipeline.process_documents(
    #     directory="./documents",
    #     chunk_size=500,
    #     chunk_overlap=50
    # )
    
    print(f"\n✓ Processed {len(result['documents'])} document chunks")
    print(f"✓ Extracted {len(result['graph_documents'])} graph structures")
    
    # Step 3: Test retrieval
    print("\n" + "=" * 60)
    print("Step 3: Testing Retrieval")
    print("=" * 60)
    
    test_question = "Who is Nonna Lucia? Did she teach anyone?"
    
    # Test graph retrieval
    print(f"\nQuestion: {test_question}")
    print("\nGraph Context:")
    graph_context = pipeline.graph_retriever(test_question)
    print(graph_context[:500] + "..." if len(graph_context) > 500 else graph_context)
    
    # Test full retrieval (graph + vector)
    print("\nFull Context (Graph + Vector):")
    full_context = pipeline.full_retriever(test_question)
    print(full_context[:500] + "..." if len(full_context) > 500 else full_context)
    
    # Step 4: Create RAG chain
    print("\n" + "=" * 60)
    print("Step 4: Creating RAG Chain")
    print("=" * 60)
    
    template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {
            "context": pipeline.full_retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | pipeline.llm
        | StrOutputParser()
    )
    
    # Step 5: Query the system
    print("\n" + "=" * 60)
    print("Step 5: Querying the System")
    print("=" * 60)
    
    questions = [
        "Who is Nonna Lucia? Did she teach anyone?",
        "What restaurants are owned by the Caruso family?",
        "What is the relationship between Giovanni and Amico?",
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = chain.invoke(question)
        print(f"A: {answer}")
    
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


def example_custom_config():
    """Example with custom configuration."""
    
    # Custom configuration
    pipeline = GraphRAGPipeline(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="your_password",
        llm_model="llama3.1",
        embedding_model="mxbai-embed-large"
    )
    
    # Process with custom chunking
    result = pipeline.process_documents(
        file_path="your_document.txt",
        chunk_size=500,      # Larger chunks
        chunk_overlap=100    # More overlap
    )
    
    # Create vector index with custom parameters
    pipeline.create_vector_index(
        search_type="hybrid",
        node_label="Document",
        text_properties=["text"],
        retrieval_kwargs={"k": 5}  # Top 5 results
    )
    
    return pipeline


def example_custom_query():
    """Example of custom query with different retrieval strategies."""
    
    pipeline = GraphRAGPipeline()
    
    # Custom graph query function
    def custom_graph_retriever(question: str) -> str:
        """Custom graph retrieval logic."""
        if pipeline.entity_chain is None:
            pipeline.setup_entity_extraction()
        
        entities = pipeline.entity_chain.invoke(question)
        result = ""
        
        for entity in entities.names:
            # Custom Cypher query
            response = pipeline.graph.query(
                """
                MATCH (e:__Entity__)
                WHERE e.id CONTAINS $entity
                OPTIONAL MATCH (e)-[r]->(related:__Entity__)
                RETURN e.id as entity, type(r) as relation, related.id as related_entity
                LIMIT 10
                """,
                {"entity": entity}
            )
            
            for record in response:
                result += f"{record['entity']} - {record['relation']} -> {record['related_entity']}\n"
        
        return result
    
    # Use custom retriever
    context = custom_graph_retriever("Who is Amico?")
    print(context)


if __name__ == "__main__":
    # Run main example
    main()
    
    # Uncomment to run other examples:
    # example_custom_config()
    # example_custom_query()

