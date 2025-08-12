import argparse
import os
import json
from datetime import datetime
from rag_pipeline import (
    extract_text_from_pdfs,
    chunk_text,
    load_or_create_embeddings_cache,
    optimized_semantic_search,
    generate_enhanced_response,
    agentic_rag_pipeline,
    log_generation_stats,
    ensure_path,
    stream_print,
    interactive_qa_session,
)


def main():
    parser = argparse.ArgumentParser(description="Agentic-RAG / RAG CLI")
    parser.add_argument("--pdf_dir", default="", help="PDF directory")
    parser.add_argument("--val", default="", help="Validation/Question JSON file path")
    parser.add_argument("--cache", default="", help="Embedding cache file")
    parser.add_argument("--stats", default="", help="Run statistics log file")
    parser.add_argument("--mode", choices=["traditional", "agentic"], default="agentic",
                        help="Run mode: 'traditional' or 'agentic'")
    parser.add_argument("--fact_check", type=int, choices=[0, 1], default=1, help="Enable fact checking (1=yes, 0=no)")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieval candidates per query")
    parser.add_argument("--similarity_threshold", type=float, default=0.55, help="RAG similarity threshold")
    parser.add_argument("--decompose_model", default="deepseek-r1:7b", help="Question decomposition model")
    parser.add_argument("--answer_model", default="qwen3:1.7b", help="Sub-question answering model")
    parser.add_argument("--synthesis_model", default="qwen3:4b", help="Final synthesis/fact-check model")
    parser.add_argument("--stream", type=int, choices=[0, 1], default=1, help="Enable streaming output (1=yes, 0=no)")
    parser.add_argument("--interactive", type=int, choices=[0, 1], default=1, help="Enable interactive Q&A mode (1=yes, 0=no)")
    args = parser.parse_args()

    ensure_path(args.val)
    ensure_path(args.cache)
    ensure_path(args.stats)

    all_pdf_texts = extract_text_from_pdfs(args.pdf_dir)
    if not all_pdf_texts:
        example_text = (
            "This is a sample text from a RAG system. RAG stands for Retrieval Enhanced Generation."
            "Agentic-RAG further adds question decomposition, retrieval planning, answer synthesis and quality assessment modules."
        )
        all_pdf_texts = [("example.pdf", example_text)]

    all_text = ""
    for pdf_file, pdf_text in all_pdf_texts:
        all_text += f"\n--- document: {pdf_file} ---\n{pdf_text}"
    text_chunks = chunk_text(all_text, n=2000, overlap=200)

    cached_embeddings = load_or_create_embeddings_cache(
        text_chunks, cache_file=args.cache, model="qwen3:0.6b"
    )

    if args.interactive:
        interactive_qa_session(text_chunks, cached_embeddings, args)
        return

    if not os.path.exists(args.val):
        example_data = [
            {"id": 1, "question": "What is the RAG system?", "answer": "..."},
            {"id": 2, "question": "How to improve the efficiency of semantic search?", "answer": "..."},
            {"id": 3, "question": "How to build an efficient agentic-RAG system?", "answer": "..."},
        ]
        with open(args.val, "w", encoding="utf-8") as f:
            json.dump(example_data, f, ensure_ascii=False, indent=2)

    with open(args.val, encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    query = data[2]["question"] if len(data) >= 3 else data[0]["question"]

    stream_callback = stream_print if args.stream else None

    if args.mode == "agentic":
        agentic_result = agentic_rag_pipeline(
            original_question=query,
            text_chunks=text_chunks,
            cached_embeddings=cached_embeddings,
            decompose_model=args.decompose_model,
            answer_model=args.answer_model,
            synthesis_model=args.synthesis_model,
            enable_fact_check=bool(args.fact_check),
            stream_callback=stream_callback,
        )

        print("\n===== Final comprehensive answer =====")
        if not args.stream:
            print(agentic_result["final_answer"])

        stats = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "agentic_rag",
            "query": query,
            "num_sub_questions": len(agentic_result["sub_questions"]),
            "sub_questions": agentic_result["sub_questions"],
            "fact_check_enabled": bool(args.fact_check),
            "fact_check_has_errors": (
                agentic_result["fact_check_result"]["has_errors"]
                if agentic_result.get("fact_check_result") else False
            ),
            "stream_enabled": bool(args.stream),
        }
        log_generation_stats(stats, args.stats)

    else:
        top_chunks = optimized_semantic_search(
            query=query,
            text_chunks=text_chunks,
            cached_embeddings=cached_embeddings,
            k=args.k,
            use_reranker=True,
            reranker_top_k=args.k,
        )
        result = generate_enhanced_response(
            query,
            top_chunks,
            similarity_threshold=args.similarity_threshold,
            model=args.synthesis_model,
            enable_fact_check=bool(args.fact_check),
            stream_callback=stream_callback,
        )

        print("\n===== Final answer =====")
        if isinstance(result, dict):
            if not args.stream:
                print(result["response"])
            stats = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "traditional_rag",
                "query": query,
                "num_chunks_found": len(top_chunks),
                "response_time": result.get("duration", 0),
                "reranker_used": True,
                "fact_check_enabled": bool(args.fact_check),
                "fact_check_has_errors": (
                    result.get("fact_check_result", {}).get("has_errors", False)
                    if result.get("fact_check_result") else False
                ),
                "stream_enabled": bool(args.stream),
            }
            log_generation_stats(stats, args.stats)
        else:
            if not args.stream:
                print(result)


if __name__ == "__main__":
    main()