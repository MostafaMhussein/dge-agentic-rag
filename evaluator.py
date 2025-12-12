#!/usr/bin/env python3
"""RAG evaluation using RAGAs metrics.

Evaluates the system using faithfulness, answer relevancy, and context precision.
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEST_QUESTIONS = [
    {
        "question": "According to the HR Bylaws, what is the standard probation period for newly appointed employees and how many times can it be extended?",
        "ground_truth": "The probation period is three months from the date of joining and may be extended only once for an additional three months, so total probation cannot exceed six months."
    }
,
    {
        "question": "How does the HR Bylaws document state that annual leave should be calculated: working days or calendar days?",
        "ground_truth": "Annual leave is calculated in calendar days, including weekends and official holidays that occur during the leave period."
    }
,
    {
        "question": "What are the disciplinary penalties for employees?",
        "ground_truth": "Disciplinary penalties can range from a verbal warning, written warning, deduction from salary, suspension from work, down-grading, up to termination of service, depending on the severity of the violation."
    }
,
    {
        "question": "What is the procedure for a Limited Tender?",
        "ground_truth": "A Limited Tender is a procurement method where a specific number of suppliers (usually not less than three) are invited to bid. It is used when the goods or services are available only from a limited number of sources or for urgent requirements."
    }
,
    {
        "question": "What is the probation period for new government employees?",
        "ground_truth": "The probation period for new government employees is typically 6 months."
    }
,
    {
        "question": "How is information classified in the security policy?",
        "ground_truth": "Information is typically classified into levels such as Public, Internal, Confidential, and Restricted (or Secret), depending on the sensitivity of the data and the impact of its disclosure."
    }
]


def main():
    print("=" * 60)
    print("RAG Evaluation")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: OPENAI_API_KEY not set")
        print("Set it in .env or export OPENAI_API_KEY=your-key")
        sys.exit(1)
    
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as e:
        print(f"\nMissing package: {e}")
        print("Install: pip install ragas datasets langchain-openai")
        sys.exit(1)
    
    from backend.rag import get_index
    from backend.agents import create_pipeline
    from backend.tracing import init_phoenix_tracing
    
    init_phoenix_tracing()
    
    print("\nInitializing pipeline...")
    try:
        index = get_index()
        pipeline, run_rag = create_pipeline(index=index)
        print("Pipeline ready")
    except Exception as e:
        print(f"Pipeline initialization failed: {e}")
        sys.exit(1)
    
    print(f"\nEvaluating {len(TEST_QUESTIONS)} questions...")
    
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    
    for i, qa in enumerate(TEST_QUESTIONS):
        question = qa["question"]
        ground_truth = qa["ground_truth"]
        
        print(f"  [{i+1}/{len(TEST_QUESTIONS)}] {question[:50]}...")
        
        try:
            answer, citations, route, contexts = run_rag(question, "")
            
            questions.append(question)
            answers.append(answer)
            contexts_list.append(contexts if contexts else ["No context"])
            ground_truths.append(ground_truth)
            
        except Exception as e:
            logger.error(f"Error: {e}")
            questions.append(question)
            answers.append(f"Error: {e}")
            contexts_list.append(["Error"])
            ground_truths.append(ground_truth)
    
    print("\nCreating dataset...")
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })
    
    print("Running RAGAs evaluation...")
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
            embeddings=embeddings,
        )
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        try:
            df = results.to_pandas()
            
            print("\nMetrics (averaged):")
            print("-" * 40)
            
            for metric in ['faithfulness', 'answer_relevancy', 'context_precision']:
                if metric in df.columns:
                    score = df[metric].mean()
                    status = "GOOD" if score >= 0.7 else "FAIR" if score >= 0.5 else "NEEDS WORK"
                    print(f"  {metric}: {score:.3f} ({status})")
            
            print("\nPer-question scores:")
            print(df[['faithfulness', 'answer_relevancy', 'context_precision']].to_string())
            
        except Exception:
            print("\nRaw results:")
            print(results)
        
    except Exception as e:
        print(f"\nEvaluation error: {e}")
        print("\nBasic results:")
        for i, (q, a) in enumerate(zip(questions, answers)):
            print(f"\nQ{i+1}: {q}")
            print(f"A: {a[:200]}...")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
