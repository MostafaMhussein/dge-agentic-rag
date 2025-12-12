"""Multi-agent RAG pipeline using CrewAI.

Implements specialized agents for query routing, optimization, 
answer generation, and validation.
"""
import logging
from typing import List, Dict, Any, Tuple, Optional

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode
from crewai import Agent, Crew, Process, Task, LLM

from .config import get_settings
from .rag import get_index, get_llm, HybridRetriever, get_reranker
from .phoenix_prompts import get_prompt
from .tracing import trace_span

logger = logging.getLogger(__name__)


def get_crewai_llm() -> LLM:
    """Create CrewAI LLM configured for Ollama."""
    settings = get_settings()
    model_string = f"ollama/{settings.ollama_model}"
    logger.info(f"Creating CrewAI LLM: {model_string}")
    
    return LLM(
        model=model_string,
        api_base=settings.ollama_base_url,
    )


class AgenticRAGPipeline:
    """Multi-agent pipeline for DGE document Q&A."""
    
    def __init__(self, index=None, nodes: Optional[List[TextNode]] = None):
        self.settings = get_settings()
        self.index = index or get_index()
        self.nodes = nodes
        
        self.crewai_llm = get_crewai_llm()
        self._setup_agents()
        
        self._retriever = HybridRetriever(self.index, nodes=nodes)
        self._reranker = get_reranker()
        self._llm = get_llm()
        
        logger.info("AgenticRAGPipeline initialized")
    
    def _setup_agents(self):
        """Initialize the agent team."""
        
        self.router = Agent(
            role="Query Router",
            goal="Classify queries accurately",
            backstory="Expert at understanding user intent for government policy queries.",
            llm=self.crewai_llm,
            verbose=False,
            allow_delegation=False,
        )
        
        self.rewriter = Agent(
            role="Query Optimizer",
            goal="Improve search queries for better retrieval",
            backstory="Specialist in information retrieval and search optimization.",
            llm=self.crewai_llm,
            verbose=False,
            allow_delegation=False,
        )
        
        self.answerer = Agent(
            role="Policy Expert",
            goal="Provide accurate, well-cited answers",
            backstory="Senior policy analyst with deep knowledge of Abu Dhabi regulations.",
            llm=self.crewai_llm,
            verbose=False,
            allow_delegation=False,
        )
        
        self.validator = Agent(
            role="Quality Checker",
            goal="Ensure responses are accurate and grounded",
            backstory="Quality assurance specialist focused on factual accuracy.",
            llm=self.crewai_llm,
            verbose=False,
            allow_delegation=False,
        )
    
    def _run_task(self, agent: Agent, description: str, expected: str) -> str:
        """Execute a single agent task."""
        task = Task(
            description=description,
            expected_output=expected,
            agent=agent,
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        
        return str(crew.kickoff()).strip()
    
    def _route(self, question: str, history: str) -> str:
        """Classify the query type."""
        with trace_span("route_query"):
            prompt = get_prompt("router", question=question, history=history or "None")
            
            result = self._run_task(
                self.router,
                prompt,
                "One word: RAG, CHITCHAT, or UNSUPPORTED"
            )
            
            route = result.upper()
            if "RAG" in route:
                return "rag"
            elif "CHITCHAT" in route:
                return "chitchat"
            return "unsupported"
    
    def _rewrite(self, question: str) -> str:
        """Optimize query for retrieval."""
        with trace_span("rewrite_query"):
            prompt = get_prompt("query-rewriter", question=question)
            
            result = self._run_task(
                self.rewriter,
                prompt,
                "A single optimized search query"
            )
            
            return result if result and len(result) > 3 else question
    
    def _retrieve(self, query: str) -> List[NodeWithScore]:
        """Execute hybrid retrieval with reranking."""
        with trace_span("retrieve"):
            query_bundle = QueryBundle(query)
            nodes = self._retriever.retrieve(query_bundle)
            
            if nodes:
                nodes = self._reranker.postprocess_nodes(nodes, query_str=query)
            
            return nodes
    
    def _build_context(self, nodes: List[NodeWithScore]) -> Tuple[str, List[Dict], List[str]]:
        """Build numbered context with citations."""
        context_parts = []
        citations = []
        raw_contexts = []
        
        for i, node in enumerate(nodes):
            filename = node.node.metadata.get("filename", "Unknown")
            text = node.node.get_content()
            
            context_parts.append(f"[{i+1}] Source: {filename}\n{text}")
            citations.append({
                "index": i + 1,
                "filename": filename,
                "score": node.score,
            })
            raw_contexts.append(text)
        
        return "\n\n".join(context_parts), citations, raw_contexts
    
    def _answer(self, question: str, context: str, history: str) -> str:
        """Generate answer with citations."""
        with trace_span("generate_answer"):
            prompt = get_prompt(
                "answer",
                question=question,
                context=context or "No relevant context found.",
                history=history or "None"
            )
            
            return self._run_task(
                self.answerer,
                prompt,
                "A comprehensive answer with citations"
            )
    
    def _handle_chitchat(self, question: str, history: str) -> str:
        """Handle casual conversation."""
        with trace_span("chitchat"):
            prompt = get_prompt("chitchat", question=question, history=history or "None")
            return self._run_task(
                self.answerer,
                prompt,
                "A friendly response"
            )
    
    def _handle_unsupported(self, question: str) -> str:
        """Handle out-of-scope queries."""
        with trace_span("unsupported"):
            prompt = get_prompt("unsupported", question=question)
            return self._run_task(
                self.answerer,
                prompt,
                "A polite explanation of scope"
            )
    
    def run(self, question: str, history_text: str = "") -> Tuple[str, List[Dict], str, List[str]]:
        """Execute the full pipeline.
        
        Returns: (answer, citations, route, raw_contexts)
        """
        with trace_span("agentic_rag_pipeline"):
            route = self._route(question, history_text)
            logger.info(f"Query routed: {route}")
            
            if route == "chitchat":
                return self._handle_chitchat(question, history_text), [], "chitchat", []
            
            if route == "unsupported":
                return self._handle_unsupported(question), [], "unsupported", []
            
            search_query = self._rewrite(question)
            logger.info(f"Search query: {search_query}")
            
            nodes = self._retrieve(search_query)
            
            if not nodes:
                return "I couldn't find relevant information in the policy documents.", [], "rag", []
            
            context_str, citations, raw_contexts = self._build_context(nodes)
            answer = self._answer(question, context_str, history_text)
            
            return answer, citations, "rag", raw_contexts


def create_pipeline(index=None, nodes=None) -> Tuple[AgenticRAGPipeline, callable]:
    """Create pipeline instance."""
    pipeline = AgenticRAGPipeline(index=index, nodes=nodes)
    return pipeline, pipeline.run


def create_crew(index=None, nodes=None):
    """Backwards-compatible wrapper."""
    pipeline, run_fn = create_pipeline(index=index, nodes=nodes)
    return pipeline, None, run_fn
