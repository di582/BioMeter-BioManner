# BioManner

A sophisticated Retrieval-Augmented Generation (RAG) system that combines traditional semantic search with agentic capabilities including question decomposition, multi-step reasoning, reranking, fact-checking, and real-time streaming output.

## 🌟 Features

### Traditional RAG Mode
- **PDF Text Extraction**: Intelligent extraction and cleaning of text from PDF documents
- **Semantic Chunking**: Smart text segmentation with configurable overlap
- **Vector Embeddings**: Efficient embedding generation with caching support
- **Cosine Similarity Search**: Fast semantic search with similarity scoring
- **Neural Reranking**: Advanced document reranking using transformer models
- **Fact Checking**: Automated fact verification and correction

### Agentic RAG Mode
- **Question Decomposition**: Breaks complex questions into manageable sub-questions
- **Multi-Step Reasoning**: Answers sub-questions individually for better accuracy
- **Answer Synthesis**: Intelligently combines sub-answers into comprehensive responses
- **Quality Assessment**: Built-in fact-checking and error correction
- **Contextual Integration**: Seamless integration of retrieved knowledge

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Documents │───▶│  Text Processing │───▶│   Embeddings    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Question  │───▶│ Question Decomp. │───▶│ Semantic Search │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Streaming Output│◀───│ Answer Synthesis │◀───│   Reranking     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│  Interactive UI │    │ Sub-Q Answering  │
└─────────────────┘    └──────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│  Fact Checking  │    │ Session Logging  │
└─────────────────┘    └──────────────────┘
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/di582/BioManner.git
   cd BioManner
   ```

2. **Install Python dependencies**
   ```bash
   pip install PyMuPDF numpy requests tqdm
   
   # Optional: For reranking capabilities
   pip install torch transformers
   ```

3. **Install and setup Ollama**
   ```bash
   # Install Ollama (visit https://ollama.ai for installation instructions)
   
   # Pull required models
   ollama pull qwen3:0.6b
   ollama pull qwen3:1.7b
   ollama pull qwen3:4b
   ollama pull deepseek-r1:7b
   ```

4. **Start Ollama server**
   ```bash
   ollama serve
   ```

## 📖 Usage

### Interactive Mode (Recommended)
```bash
# Start interactive session with streaming
python app.py --interactive 1 --mode agentic --stream 1

# Interactive traditional RAG
python app.py --interactive 1 --mode traditional --stream 1
```

### Single Question Mode
```bash
# Traditional RAG Mode
python app.py --mode traditional --pdf_dir /path/to/pdfs --k 5

# Agentic RAG Mode  
python app.py --mode agentic --pdf_dir /path/to/pdfs --fact_check 1
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--interactive` | `0` | Enable interactive mode (1=yes, 0=no) |
| `--stream` | `1` | Enable streaming output (1=yes, 0=no) |
| `--pdf_dir` | `./pdfs` | Directory containing PDF documents |
| `--val` | `./qa.json` | JSON file with questions for testing |
| `--cache` | `./embeddings_cache.pkl` | Embedding cache file |
| `--mode` | `traditional` | Mode: `traditional` or `agentic` |
| `--fact_check` | `1` | Enable fact checking (1=yes, 0=no) |
| `--k` | `5` | Number of retrieval candidates |
| `--similarity_threshold` | `0.55` | Minimum similarity threshold |
| `--decompose_model` | `deepseek-r1:7b` | Question decomposition model |
| `--answer_model` | `qwen3:1.7b` | Sub-question answering model |
| `--synthesis_model` | `qwen3:4b` | Final synthesis model |

### 💬 Interactive Commands

Once in interactive mode, you can use these commands:
- Type your question and press Enter
- `help` - Show available commands
- `quit`, `exit`, or `q` - Exit the session
- `Ctrl+C` - Force exit

### Python API Usage

```python
from rag_pipeline import (
    extract_text_from_pdfs,
    chunk_text,
    load_or_create_embeddings_cache,
    agentic_rag_pipeline,
    interactive_qa_session
)

# Extract and process documents
pdf_texts = extract_text_from_pdfs("./pdfs")
all_text = "\n".join([text for _, text in pdf_texts])
chunks = chunk_text(all_text, n=2000, overlap=200)

# Create embeddings
embeddings = load_or_create_embeddings_cache(chunks)

# Run Agentic RAG
result = agentic_rag_pipeline(
    original_question="How does machine learning work?",
    text_chunks=chunks,
    cached_embeddings=embeddings,
    stream_callback=lambda token: print(token, end='', flush=True)
)

print(result["final_answer"])

# Start interactive session
interactive_qa_session(
    text_chunks=chunks,
    cached_embeddings=embeddings,
    mode="agentic",
    enable_streaming=True
)
```

## 🔧 Configuration

### Model Configuration
The system supports multiple Ollama models. Configure them in the command line or modify the defaults in the code:

- **Embedding Model**: Generates vector representations of text (`qwen3:0.6b`)
- **Decomposition Model**: Breaks down complex questions (`deepseek-r1:7b`)
- **Answer Model**: Answers individual sub-questions (`qwen3:1.7b`)
- **Synthesis Model**: Combines answers and performs fact-checking (`qwen3:4b`)

### Reranking Setup
For neural reranking, place a compatible transformer model in the `./reranker` directory:
```bash
# Example: Download a reranking model
huggingface-cli download BAAI/bge-reranker-base ./reranker
```

### Streaming Configuration
Streaming output provides real-time feedback during response generation:
```python
# Enable streaming with custom callback
def my_stream_callback(token):
    print(f"[TOKEN]: {token}", end='', flush=True)

result = agentic_rag_pipeline(
    question="Your question here",
    stream_callback=my_stream_callback
)
```
