"""
Utility script to check and manage Ollama models for GraphRAG Pipeline.
Use this script to inspect available models and pull missing ones.
"""

from graphrag_pipeline import GraphRAGPipeline


def main():
    """Main function to check and manage Ollama models."""
    print("=" * 70)
    print("Ollama Model Manager for GraphRAG Pipeline")
    print("=" * 70)
    
    # Check if Ollama is available
    print("\n1. Checking Ollama availability...")
    if not GraphRAGPipeline.check_ollama_available():
        print("✗ Ollama is not available or not running")
        print("\nPlease ensure:")
        print("  1. Ollama is installed from https://ollama.ai")
        print("  2. Ollama service is running")
        print("  3. Try running: ollama list")
        return
    
    print("✓ Ollama is available\n")
    
    # List all available models
    print("2. Current Ollama models:")
    print("-" * 70)
    models = GraphRAGPipeline.list_ollama_models()
    
    if models:
        print(f"{'Model Name':<30} {'ID':<20} {'Size':<15}")
        print("-" * 70)
        for model in models:
            name = model.get('name', 'N/A')
            model_id = model.get('id', 'N/A')[:20] if model.get('id') else 'N/A'
            size = model.get('size', 'N/A')
            print(f"{name:<30} {model_id:<20} {size:<15}")
    else:
        print("  No models found")
    
    # Check for required models
    print("\n" + "=" * 70)
    print("3. Required models for GraphRAG Pipeline:")
    print("-" * 70)
    
    required_models = ["llama3.1", "mistral"]
    
    for model in required_models:
        status = GraphRAGPipeline.ensure_ollama_models(
            required_models=[model],
            auto_pull=False,
            check_only=True
        )
        is_available = status.get(model, False)
        icon = "✓" if is_available else "✗"
        print(f"  {icon} {model:<25} {'Available' if is_available else 'Missing'}")
    
    # Option to pull missing models
    print("\n" + "=" * 70)
    print("4. Pull missing models?")
    print("-" * 70)
    
    # Check which are missing
    all_status = GraphRAGPipeline.ensure_ollama_models(
        required_models=required_models,
        auto_pull=False,
        check_only=True
    )
    missing = [model for model, status in all_status.items() if not status]
    
    if missing:
        print(f"\nMissing models: {', '.join(missing)}")
        response = input("\nWould you like to pull missing models? (y/n): ").strip().lower()
        
        if response == 'y':
            print("\nPulling missing models...")
            for model in missing:
                print(f"\n{'='*70}")
                print(f"Pulling: {model}")
                print("="*70)
                GraphRAGPipeline.pull_ollama_model(model)
        else:
            print("\nTo pull models manually, run:")
            for model in missing:
                print(f"  ollama pull {model}")
    else:
        print("\n✓ All required models are available!")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()

