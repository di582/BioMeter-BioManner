import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import os
import re
import json
import time
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import fitz
import numpy as np
import requests
from tqdm import tqdm


try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


class TimingContext:
    def __init__(self, description):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"Start {self.description}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        print(f"Finish {self.description}, time consuming: {self.end_time - self.start_time:.2f} second")
        return False


_reranker_model = None
_reranker_tokenizer = None


def clean_pdf_text(text: str) -> str:
    with TimingContext("PDF text cleaning"):
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef.,;:!?()[\]{}"\'~@#$%^&*+=|\\/<>-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if (len(line) < 10 and
                (re.match(r'^\d+$', line) or
                 re.match(r'^Page\s+\d+', line, re.I) or
                 re.match(r'^\d+\s*$', line) or
                 re.match(r'^[A-Z\s]{3,}$', line))):
                continue
            cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)

        refs_patterns = [
            r'\n\s*(?:References|REFERENCES|Bibliography)\s*\n',
            r'\n\s*\d+\.\s*(?:References)',
            r'\n\s*\[\d+\]\s*[A-Z]'
        ]
        for pattern in refs_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                text = text[:m.start()]
                break

        figure_table_patterns = [
            r'(?:Figure|Fig|Table|Tab)\s*\d+[:\s]*[^\n]*\n?',
            r'\n\s*\d+\.\d+\s*[^\n]*\n',
        ]
        for pattern in figure_table_patterns:
            text = re.sub(pattern, '\n', text, flags=re.IGNORECASE)

        copyright_patterns = [
            r'©\s*\d{4}[^\n]*\n?',
            r'Copyright[^\n]*\n?',
            r'All rights reserved[^\n]*\n?',
            r'doi:\s*[^\s\n]+',
            r'DOI:\s*[^\s\n]+',
            r'ISSN[:\s]*[0-9-]+',
            r'ISBN[:\s]*[0-9-]+',
        ]
        for pattern in copyright_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d+\)', '', text)
        text = re.sub(r'[∑∫∂∆∇∞±≤≥≠≈∈∉∪∩⊂⊃∀∃∧∨¬→↔]+\s*', ' ', text)
        text = re.sub(r'\n\s*(\d+\.?\d*\.?\d*)\s+([A-Z][^\n]*)\n', r'\n\1 \2\n', text)

        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if (len(para) < 20 or
                len(para.split()) < 3 or
                re.match(r'^[A-Z\s]+$', para) or
                re.match(r'^\d+[\s\.]*$', para)):
                continue
            para = re.sub(r'[.,;:!?]{2,}', '.', para)
            cleaned_paragraphs.append(para)

        text = '\n\n'.join(cleaned_paragraphs)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()


def load_reranker_model(model_path="reranker"):
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is not None and _reranker_tokenizer is not None:
        return _reranker_model, _reranker_tokenizer

    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        print("Transformers/torch is not installed, skipping reranking model loading.")
        return None, None

    with TimingContext("Loading the reranking model"):
        try:
            print(f"Load reranking model from path: {model_path}")
            _reranker_tokenizer = AutoTokenizer.from_pretrained(model_path)
            _reranker_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            _reranker_model.eval()
            if torch and torch.cuda.is_available():
                _reranker_model = _reranker_model.cuda()
                print("The reranking model has been moved to the GPU")
            else:
                print("Run the reranking model using the CPU")
        except Exception as e:
            print(f"Failed to load re-ranking model: {e}, continue to use similarity ranking.")
            _reranker_model = None
            _reranker_tokenizer = None
        return _reranker_model, _reranker_tokenizer


def rerank_documents(query, documents, model_path="reranker", top_k=5):
    with TimingContext("Rerank documents"):
        model, tokenizer = load_reranker_model(model_path)
        if model is None or tokenizer is None or not documents:
            if not documents:
                return []
            uniform = 1.0 / len(documents)
            return [(idx, uniform, content) for idx, content in documents]

        pairs = [[query, content] for idx, content in documents]
        batch_size = 8
        all_scores = []

        try:
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                inputs = tokenizer(
                    batch_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
                )
                if torch and torch.cuda.is_available() and model.device.type == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    scores = outputs.logits.squeeze(-1).detach().cpu().numpy()
                    if len(scores.shape) == 0:
                        scores = [float(scores)]
                all_scores.extend(scores)

            all_scores = np.array(all_scores, dtype=np.float32)
            if torch:
                norm = torch.softmax(torch.tensor(all_scores), dim=0).numpy()
            else:
                ex = np.exp(all_scores - np.max(all_scores))
                norm = ex / (np.sum(ex) + 1e-9)

            reranked = []
            for i, (doc_idx, doc_content) in enumerate(documents):
                reranked.append((doc_idx, float(norm[i]), doc_content))
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked[: top_k or len(reranked)]
        except Exception as e:
            print(f"Reranking error: {e}, falling back to initial sorting.")
            uniform = 1.0 / len(documents)
            return [(idx, uniform, content) for idx, content in documents]


def extract_text_from_pdfs(pdf_directory: str):
    with TimingContext("PDF Text Extraction"):
        all_texts = []
        if not os.path.exists(pdf_directory):
            print(f"Warning: PDF directory not found {pdf_directory}")
            return all_texts

        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"Warning: No PDF files found in directory {pdf_directory}")
            return all_texts

        for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            try:
                mypdf = fitz.open(pdf_path)
                pdf_text = ""
                for page in mypdf:
                    pdf_text += page.get_text("text")
                cleaned_text = clean_pdf_text(pdf_text)
                all_texts.append((pdf_file, cleaned_text))
                print(f"Successfully extracted and cleaned {pdf_file}, original: {len(pdf_text)} -> after cleaning: {len(cleaned_text)}")
            except Exception as e:
                print(f"An error occurred while extracting text from {pdf_file}: {e}")
        return all_texts


def chunk_text(text: str, n: int, overlap: int):
    with TimingContext("Text block processing"):
        sentences, cur = [], ""
        for ch in text:
            cur += ch
            if ch in ['。', '？', '！', '.', '?', '!']:
                sentences.append(cur)
                cur = ""
        if cur:
            sentences.append(cur)

        chunks, cur = [], ""
        for s in sentences:
            if len(cur) + len(s) <= n or not cur:
                cur += s
            else:
                chunks.append(cur)
                words = cur.split()
                if len(words) > max(1, overlap // 10):
                    overlap_text = ' '.join(words[-max(1, overlap // 10):])
                else:
                    overlap_text = cur[-overlap:] if len(cur) > overlap else cur
                cur = overlap_text + s
        if cur:
            chunks.append(cur)
        return chunks


def create_single_embedding(text: str, model="qwen3:0.6b"):
    url = "http://localhost:11434/api/embeddings"
    max_text_length = 8000
    if len(text) > max_text_length:
        text = text[:max_text_length]

    payload = {"model": model, "prompt": text}
    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json().get("embedding", [])
    except requests.RequestException as e:
        print(f"Generate embedding error: {e}")
        if len(text) > 1000:
            half = len(text) // 2
            e1 = create_single_embedding(text[:half], model)
            e2 = create_single_embedding(text[half:], model)
            if e1 and e2 and len(e1) == len(e2):
                return [(a + b) / 2 for a, b in zip(e1, e2)]
        return []


def create_embeddings_batch(texts, model="qwen3:0.6b", batch_size=8):
    with TimingContext("Batch generate embeddings"):
        all_embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Generating batch embeddings"):
            batch_texts = texts[i * batch_size: min((i + 1) * batch_size, len(texts))]
            batch_embeddings = []
            with ThreadPoolExecutor(max_workers=min(len(batch_texts), 4)) as ex:
                futures = [ex.submit(create_single_embedding, t, model) for t in batch_texts]
                for fut in futures:
                    batch_embeddings.append(fut.result())
            all_embeddings.extend(batch_embeddings)
            time.sleep(0.2)
        return all_embeddings


def compute_cosine_similarities(q_emb, chunk_embs):
    with TimingContext("Calculate cosine similarity"):
        q = np.array(q_emb, dtype=np.float32)
        X = np.array(chunk_embs, dtype=np.float32)
        if len(X.shape) != 2 or X.shape[1] != len(q):
            raise ValueError("Embedding dimensions do not match or are empty.")
        dots = np.dot(X, q)
        qn = np.linalg.norm(q) + 1e-12
        Xn = np.linalg.norm(X, axis=1) + 1e-12
        return dots / (Xn * qn)


def load_or_create_embeddings_cache(text_chunks, cache_file="embeddings_cache.pkl", model="qwen3:0.6b"):
    with TimingContext("Load/create embedding cache"):
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                if len(cached) == len(text_chunks):
                    print(f"Loaded cache embedding: {len(cached)}")
                    return cached
                else:
                    print("The number of caches does not match the number of chunks, regenerating the embedding.")
            except Exception as e:
                print(f"Cache load failed: {e}, will be regenerated.")

        print("Start creating embedding for text blocks...")
        embeddings = create_embeddings_batch(text_chunks, model=model)
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Saved {len(embeddings)} embeddings to {cache_file}")
        return embeddings


def optimized_semantic_search(query, text_chunks, cached_embeddings=None, k=5,
                              model="qwen3:0.6b", use_reranker=True, reranker_top_k=None):
    with TimingContext("Semantic search (including reranking)"):
        q_emb = create_single_embedding(query, model)
        if not q_emb:
            print("The query embedding is empty, returning empty results.")
            return []

        chunk_embs = cached_embeddings or create_embeddings_batch(text_chunks, model=model)
        sims = compute_cosine_similarities(q_emb, chunk_embs)
        initial = [(i, float(sims[i]), text_chunks[i]) for i in range(len(text_chunks))]
        initial.sort(key=lambda x: x[1], reverse=True)

        cand_num = min(k * 3, len(initial))
        candidates = initial[:cand_num]

        if use_reranker and len(candidates) > 1:
            rerank_in = [(idx, chunk) for idx, _, chunk in candidates]
            reranked = rerank_documents(query, rerank_in, top_k=reranker_top_k or k)
            return reranked[:k]
        return candidates[:k]


def generate_response(system_prompt, user_message, model="qwen3:0.6b", stream_callback=None):
    url = "http://localhost:11434/api/generate"
    full_prompt = f"{system_prompt}\n\n{user_message}"
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "num_predict": 4096,
        "stop": ["<End of answer>"],
    }
    print(f"Starting generating responses (model: {model})...")
    start = time.time()
    try:
        resp_text = ""
        with requests.post(url, json=payload, timeout=180, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                token = data.get("response", "")
                resp_text += token
                if stream_callback and token:
                    stream_callback(token)
                if data.get("done", False):
                    break
        dur = time.time() - start
        print(f"\nGeneration completed, took {dur:.2f}s")
        return {"response": resp_text.strip() or "(empty response)", "duration": dur}
    except requests.RequestException as e:
        dur = time.time() - start
        msg = f"Error generating response: {e}"
        print(msg)
        return {"response": msg, "duration": dur}


def fact_check_and_correct(answer, original_question, model="qwen3:4b", stream_callback=None):
    with TimingContext("Fact-checking and error correction"):
        system_prompt = """You are a professional fact-checker with extensive domain knowledge and a particular skill in identifying and correcting factual errors in technical documentation.

Your task is:
1. Carefully review the provided answers and identify any possible factual errors.
2. Focus on technical parameter errors, process condition errors, equipment specification errors, theoretical principle errors, standard specification errors, chronological errors, and causal errors.
3. Key areas of review include biotechnology, chemical engineering, mechanical engineering, computer science, basic principles of physics and chemistry.
4. Correction Principles:
- Maintain the overall structure and style of the original answer.
- Correct only content that is genuinely incorrect and avoid unnecessary changes.
- Use accurate technical terminology and standard values when correcting.
- Ensure that the corrected content is logically consistent and coherent.
5. Output Format:
If an error is found, output:
[Errors Found]
Error 1: [Details of the error and its location]
Correct: [Corrected statement]
[Corrected Complete Answer]
[Provide the complete corrected answer]

If no obvious errors are found, output:
[Check Results]
No obvious factual errors are found. The original answer is generally accurate.
[Confirmed Answer]
[Original Answer Content]"""

        user_message = f"""Please fact-check and correct the following answers:

Original question:
{original_question}

Answer to be checked:
{answer}

Please carefully check whether there are any factual errors in the answer, especially errors in technical parameters, process conditions, theoretical principles, etc., and output the inspection results in the specified format."""

        try:
            result = generate_response(system_prompt, user_message, model, stream_callback)
            text = result["response"].strip()
            if "[Errors Found]" in text and "[Corrected Complete Answer]" in text:
                corrected = text.split("[Corrected Complete Answer]", 1)[-1].strip()
                return {
                    "has_errors": True,
                    "fact_check_details": text,
                    "corrected_answer": corrected or answer,
                    "original_answer": answer,
                }
            else:
                return {
                    "has_errors": False,
                    "fact_check_details": text,
                    "corrected_answer": answer,
                    "original_answer": answer,
                }
        except Exception as e:
            return {
                "has_errors": False,
                "fact_check_details": f"Fact check failed: {e}",
                "corrected_answer": answer,
                "original_answer": answer,
            }


def decompose_question(original_question, model="deepseek-r1:7b", stream_callback=None):
    with TimingContext("Question decomposition"):
        system_prompt = """You are a professional question decomposition expert. Your task is to break down complex user questions into multiple, more specific and easier-to-answer sub-questions.

Decomposition Principles:
1. Sub-questions must focus on the core concepts of the original question, remain highly relevant, and avoid introducing irrelevant content.
2. Each sub-question should be specific and answerable through search documents or models, avoiding being too general or abstract.
3. There should be a logical progression between sub-questions, typically from "what → what → how → how to optimize."
4. The combined answers to the sub-questions should fully answer the original question.
5. It is recommended to decompose the question into 3-5 sub-questions.

Please output the decomposition results strictly in the following JSON format:
{
"sub_questions": [
"Sub-question 1",
"Sub-question 2",
"Sub-question 3",
"Sub-question 4"
],
"reasoning": "Decomposition strategy: Start with the conceptual definition and gradually move on to the components, working mechanism, and implementation strategy."
}

Do not include any additional text or explanation."""

        user_message = f"Original question: {original_question}\n\nPlease break this question down into multiple sub-questions that follow a logical progression."
        result = generate_response(system_prompt, user_message, model, stream_callback)
        text = result["response"]
        try:
            s, e = text.find("{"), text.rfind("}") + 1
            obj = json.loads(text[s:e])
            subs = obj.get("sub_questions", []) or [original_question]
            reasoning = obj.get("reasoning", "")
            return subs, reasoning
        except Exception as e:
            print(f"Failed to parse the decomposition result: {e}")
            return [original_question], "Decomposition failed, use the original problem"


def answer_sub_question(sub_question, relevant_chunks, model="qwen3:1.7b", stream_callback=None):
    with TimingContext(f"Answer sub-questions: {sub_question[:50]}"):
        system_prompt = """You are a professional knowledge assistant. You are expected to answer user questions, but please note the following requirements:

Answer Requirements:
1. Answer based on the background information provided, incorporating your expertise.
2. Answer accurately, specifically, and logically, providing in-depth explanation.
3. If the background information is insufficient to answer the question, supplement with relevant information based on your expertise.
4. Maintain completeness and professionalism in your answer, but avoid being overly verbose.
5. Keep your answer between 300-600 words to ensure it is comprehensive.
6. Avoid quotations such as "according to the document," "references show," or "document fragments" in your final answer.
7. Integrate the information in a natural and fluent manner, as if you already possess this knowledge.

Answer Structure:
- Directly answer the core question.
- Provide necessary background information and detailed explanation.
- Incorporate specific examples or technical details naturally, if necessary.
- Ensure your answer is complete and self-sufficient for the sub-question.

Please provide your answer directly in a professional and natural tone, without revealing the source of the information."""

        if relevant_chunks:
            doc = "\n\n".join([f"Background information{i+1}:\n{c[2]}" for i, c in enumerate(relevant_chunks)])
            user_message = f"Background information:\n{doc}\n\nQuestion: {sub_question}\n\nPlease provide a thorough and natural answer based on this background information and your expertise. Remember: do not refer to background information or documents in your answer; answer as if you already have this knowledge."
        else:
            user_message = f"Question: {sub_question}\n\nPlease provide a detailed answer based on your expertise."
        result = generate_response(system_prompt, user_message, model, stream_callback)
        return result["response"].strip()


def synthesize_final_answer(original_question, sub_qa_pairs, model="qwen3:4b", stream_callback=None):
    with TimingContext("Comprehensive final answer"):
        system_prompt = """You are a professional knowledge integration expert, specializing in integrating multiple related knowledge points into precise answers to specific questions.

Core Principles:
1. Strictly Align to the Original Question: The final answer must directly and completely answer the original question and must not stray from the topic.
2. Information Filtering and Focusing: Only extract information directly relevant to the original question and resolutely exclude irrelevant content.
3. Logical Reconstruction: Reorganize the content according to the inherent logic of the original question to form a coherent knowledge system.
4. Deep Integration: Demonstrate a deep understanding of the original question and provide an insightful and comprehensive answer.
5. Natural Expression: Answer in a fluent and natural tone, avoiding procedural terms such as "sub-questions," "integration," and "synthesis."
6. Hidden Traces: Do not reveal the process of information integration; it should appear as if the knowledge is already fully understood.

Quality Standards:
- Every sentence in the answer must serve the original question
- Ensure the completeness of the answer: Cover all key aspects of the original question
- Ensure the accuracy of the answer: Do not distort or overextend existing information
- Ensure your answer is practical: It provides practical value to the questioner
- Ensure your answer is natural: It reads like a coherent, professional response

Answer Format:
- Start by directly addressing the core of the original question
- In the middle, systematically expand on all aspects of the original question
- Conclude with a concise and powerful summary of key points

Maintain a professional yet natural tone throughout."""

        knowledge = ""
        for i, (sq, sa) in enumerate(sub_qa_pairs, 1):
            knowledge += f"\n##### Related knowledge points {i}\n**About: {sq}**\n**Content: **{sa.strip()}\n"
        user_message = f"""Task Objective:
Please generate a professional answer to the original question based on the relevant knowledge provided.

Original Question:
{original_question}

Core Task:
Your task is to answer "{original_question}" professionally.

Important Requirements:
- Your final answer must fully address the original question: "{original_question}"
- Please carefully analyze the specific requirements and keywords of the original question
- Extract information from relevant knowledge and reorganize it into a format that directly answers the original question
- Avoid using terms such as "integration," "synthesis," "sub-questions," or "related knowledge" in your answer
- Express this information naturally, as if it were your complete expertise
- If some information is not particularly relevant to the original question, it may be omitted
- If you find that information is insufficient to fully answer the original question, please clearly indicate this

Explanation Strategy:
1. Answer the original question: "{original_question}" directly and professionally
2. Naturally develop each key point according to the logical requirements of the question
3. Develop a clear, structured, and well-organized answer
4. Maintain fluency and professionalism
5. Avoid any statements that reveal the source or synthesis of information

Related Knowledge: {knowledge}

Now, please focus on answering the original question: "{original_question}."
Please generate your final answer (recommended 1,000 words or more, with depth and practicality, and a natural and professional tone):"""

        result = generate_response(system_prompt, user_message, model, stream_callback)
        return result["response"].strip()


def agentic_rag_pipeline(original_question, text_chunks, cached_embeddings,
                         decompose_model="deepseek-r1:7b",
                         answer_model="qwen3:1.7b",
                         synthesis_model="qwen3:4b",
                         enable_fact_check=True,
                         stream_callback=None):
    with TimingContext("Agentic-RAG complete process"):
        subs, reasoning = decompose_question(original_question, decompose_model, stream_callback)
        sub_qa_pairs = []

        for subq in subs:
            rel = optimized_semantic_search(
                subq, text_chunks, cached_embeddings, k=3, use_reranker=True
            )
            ans = answer_sub_question(subq, rel, answer_model, stream_callback)
            sub_qa_pairs.append((subq, ans))

        final_answer = synthesize_final_answer(original_question, sub_qa_pairs, synthesis_model, stream_callback)
        fact_check_result = None
        if enable_fact_check:
            fact_check_result = fact_check_and_correct(final_answer, original_question, synthesis_model, stream_callback)
            final_answer = fact_check_result["corrected_answer"]

        return {
            "original_question": original_question,
            "decomposition_reasoning": reasoning,
            "sub_questions": subs,
            "sub_qa_pairs": sub_qa_pairs,
            "final_answer": final_answer,
            "fact_check_result": fact_check_result,
        }


def generate_enhanced_response(query, top_chunks, similarity_threshold=0.6,
                               model="qwen3:4b", enable_fact_check=True, stream_callback=None):
    with TimingContext("Generate enhanced response"):
        relevant = [(i, s, c) for i, (i2, s, c) in enumerate(top_chunks) if s >= similarity_threshold]

        if not top_chunks:
            system_prompt = """You are a professional AI assistant, skilled at providing detailed answers, leveraging the power of the Qwen3:4b large language model.

Please answer user questions according to the following requirements:
1. Comprehensiveness: Provide a complete, in-depth answer that covers all aspects of the question.
2. Clear logic: Use a hierarchical structure to sequentially expand on each knowledge point.
3. Detailed explanation: Provide a thorough explanation of key concepts, processes, or mechanisms.
4. Examples: Use concrete examples or metaphors to make complex concepts easier to understand.
5. Optimized formatting: Use appropriate headings, bullet points, and paragraph breaks to enhance readability, depending on the nature of the content.
6. Expertise and accuracy: Ensure accurate information and avoid vague or misleading statements.
7. Sufficient length: Provide an answer long enough to fully cover all aspects of the question.

Your answer should be detailed and authoritative, at least 800 words long, demonstrating the full knowledge of your advanced language model.
Expand on relevant knowledge points whenever possible, but always stay focused on the core of the question.
Be sure to provide sufficient relevant information to ensure the user has a comprehensive and in-depth understanding."""

            result = generate_response(system_prompt, f"Question: {query}", model, stream_callback)
            answer = result["response"]
            if enable_fact_check:
                fc = fact_check_and_correct(answer, query, model, stream_callback)
                return {"response": fc["corrected_answer"], "duration": result["duration"], "fact_check_result": fc}
            return {"response": answer, "duration": result["duration"], "fact_check_result": None}

        user_prompt = "Reference Information:\n"
        if relevant:
            for i, (idx, score, chunk) in enumerate(relevant):
                user_prompt += f"[Text Block {i+1}] (Relevance: {score:.4f})\n{chunk}\n\n"
        else:
            user_prompt += "No highly relevant reference texts were found.\n\n"
        user_prompt += f"Question: {query}\n\nPlease follow the system's instructions, combining reference information with your own knowledge to provide a comprehensive, detailed, and accurate answer. Ensure your answer is informative and long enough (at least 800 words) to demonstrate your expertise as a leading language model."

        system_prompt = """You are a professional knowledge assistant, based on the qwen3:4b model, specializing in providing extremely comprehensive, detailed, and accurate answers. You will receive a user question and some reference text blocks related to the question.
Please strictly follow the following guidelines when answering:

1. Analysis and Integration:
- Thoroughly analyze all reference text blocks to identify key information that is highly relevant to the question.
- Integrate information from multiple text blocks to form a complete and coherent body of knowledge.
- When resolving conflicting information, prioritize more authoritative and specific content.

2. Content Expansion:
- Substantially expand and enrich the content of the reference text blocks.
- Supplement necessary background information, definitions, theoretical frameworks, etc.
- Add additional knowledge related to the topic to enrich your answer.
- Provide multi-angle analysis and multi-layered interpretation.

3. In-Depth Integration:
- If the reference text block contains complete and relevant information, primarily use this information to construct a detailed answer and expand on the relevant details.
- If the reference text block contains partial relevant information, seamlessly integrate this information with your own expertise to construct a more comprehensive answer.
- If the reference text block does not contain sufficient information, primarily rely on your own knowledge base, but still refer to the structure and style of the relevant text.

4. Completeness and Detail:
- Ensure your answer covers all aspects and dimensions of the question.
- Provide detailed explanation and elaboration on key concepts.
- Include specific examples, application scenarios, or case studies. Answers should be sufficiently long, at least 800 words, ensuring comprehensive and in-depth information.

5. Professional Accuracy:
- Use domain-specific terminology to demonstrate academic depth.
- Maintain precise expression and avoid vague or potentially misleading terms.
- Cite data, research, or authoritative opinions where necessary to enhance credibility.

In all cases, ensure your answer is extremely detailed, professional, and directly addresses the user's question. Your answer should demonstrate the full knowledge and reasoning capabilities of the qwen3:4b model."""

        result = generate_response(system_prompt, user_prompt, model, stream_callback)
        answer = result["response"]
        if enable_fact_check:
            fc = fact_check_and_correct(answer, query, model, stream_callback)
            return {"response": fc["corrected_answer"], "duration": result["duration"], "fact_check_result": fc}
        return {"response": answer, "duration": result["duration"], "fact_check_result": None}


def log_generation_stats(stats, log_file="generation_stats.jsonl"):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(stats, ensure_ascii=False) + "\n")


def ensure_path(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def stream_print(token):
    print(token, end='', flush=True)


def interactive_qa_session(text_chunks, cached_embeddings, args):
    print("\n" + "="*60)
    print("Interactive Q&A Session Started")
    print("Type 'quit', 'exit', or 'q' to end the session")
    print("Type 'help' for available commands")
    print("="*60)
    
    session_stats = []
    
    while True:
        try:
            query = input("\nEnter your question: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("Ending interactive session...")
                break
                
            if query.lower() == 'help':
                print("\nAvailable commands:")
                print("  help - Show this help message")
                print("  quit/exit/q - End the session")
                print("  Any other input - Ask a question")
                continue
                
            print(f"\nProcessing question: {query}")
            print("-" * 50)
            
            stream_callback = stream_print if args.stream else None
            start_time = datetime.now()
            
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
                
                if not args.stream:
                    print("\n" + "="*50)
                    print("ANSWER:")
                    print("="*50)
                    print(agentic_result["final_answer"])
                
                stats = {
                    "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
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
                    "interactive_mode": True,
                }
                
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
                
                if not args.stream and isinstance(result, dict):
                    print("\n" + "="*50)
                    print("ANSWER:")
                    print("="*50)
                    print(result["response"])
                
                stats = {
                    "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": "traditional_rag",
                    "query": query,
                    "num_chunks_found": len(top_chunks),
                    "response_time": result.get("duration", 0) if isinstance(result, dict) else 0,
                    "reranker_used": True,
                    "fact_check_enabled": bool(args.fact_check),
                    "fact_check_has_errors": (
                        result.get("fact_check_result", {}).get("has_errors", False)
                        if isinstance(result, dict) and result.get("fact_check_result") else False
                    ),
                    "stream_enabled": bool(args.stream),
                    "interactive_mode": True,
                }
            
            session_stats.append(stats)
            log_generation_stats(stats, "generation_stats.jsonl")
            
            print("\n" + "-"*50)
            print("Question answered. Ask another question or type 'quit' to exit.")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Ending session...")
            break
        except Exception as e:
            print(f"\nError processing question: {e}")
            continue
    
    print(f"\nSession completed. Processed {len(session_stats)} questions.")
    return session_stats