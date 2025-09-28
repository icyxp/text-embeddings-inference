#!/usr/bin/env python3
"""
Qwen3 Reranker Template Utilities

This module provides template formatting functions for Qwen3 reranker models,
ensuring compatibility with the official Qwen3-Reranker format.

The template format matches exactly what's used in the official models:
- tomaarsen/Qwen3-Reranker-0.6B-seq-cls
- tomaarsen/Qwen3-Reranker-4B-seq-cls
- tomaarsen/Qwen3-Reranker-8B-seq-cls
"""

import torch
from typing import List, Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def format_qwen3_reranker_input(
    instruction: Optional[str],
    query: str,
    document: str
) -> str:
    """
    Format input for Qwen3 reranker using the official template.
    
    This function creates the exact format expected by Qwen3 reranker models,
    including the system prompt, user input structure, and assistant prompt.
    
    Args:
        instruction: Task instruction (if None, uses default)
        query: The search query
        document: The document to be ranked
        
    Returns:
        Formatted string ready for tokenization
    """
    prefix = (
        '<|im_start|>system\n'
        'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
        'Note that the answer can only be "yes" or "no".<|im_end|>\n'
        '<|im_start|>user\n'
    )
    
    suffix = (
        '<|im_end|>\n'
        '<|im_start|>assistant\n'
        '<think>\n\n</think>\n\n'
    )
    
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    
    formatted_input = (
        f"{prefix}"
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
        f"{suffix}"
    )
    
    return formatted_input


def format_batch_inputs(
    instruction: Optional[str],
    queries: List[str],
    documents: List[str]
) -> List[str]:
    """
    Format multiple query-document pairs for batch processing.
    
    Args:
        instruction: Task instruction (same for all pairs)
        queries: List of queries
        documents: List of documents
        
    Returns:
        List of formatted strings
    """
    if len(queries) != len(documents):
        raise ValueError("Number of queries must match number of documents")
    
    return [
        format_qwen3_reranker_input(instruction, query, doc)
        for query, doc in zip(queries, documents)
    ]


def format_cross_encoder_pairs(
    instruction: Optional[str],
    queries: List[str],
    documents: List[str]
) -> List[List[str]]:
    """
    Format inputs for sentence-transformers CrossEncoder format.
    
    Args:
        instruction: Task instruction
        queries: List of queries
        documents: List of documents
        
    Returns:
        List of [query_formatted, document_formatted] pairs
    """
    if len(queries) != len(documents):
        raise ValueError("Number of queries must match number of documents")
    
    prefix = (
        '<|im_start|>system\n'
        'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
        'Note that the answer can only be "yes" or "no".<|im_end|>\n'
        '<|im_start|>user\n'
    )
    
    suffix = (
        '<|im_end|>\n'
        '<|im_start|>assistant\n'
        '<think>\n\n</think>\n\n'
    )
    
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    
    pairs = []
    for query, doc in zip(queries, documents):
        query_part = f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        doc_part = f"<Document>: {doc}{suffix}"
        pairs.append([query_part, doc_part])
    
    return pairs


class Qwen3RerankerWrapper:
    """
    Wrapper class for Qwen3 reranker models with automatic template formatting.
    
    This class handles both converted sequence classification models and
    original causal language models with proper template formatting.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        model_type: str = "sequence_classification",
        max_length: int = 8192,
        device: Optional[str] = None
    ):
        """
        Initialize the Qwen3 reranker wrapper.
        
        Args:
            model_name_or_path: Path to model or HuggingFace model name
            model_type: Either "sequence_classification" or "causal_lm"
            max_length: Maximum sequence length
            device: Device to load model on (auto-detected if None)
        """
        self.model_name = model_name_or_path
        self.model_type = model_type
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left"
        )
        
        # Load model based on type
        if model_type == "sequence_classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path
            ).to(self.device).eval()
            self._predict_fn = self._predict_seq_cls
        elif model_type == "causal_lm":
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path
            ).to(self.device).eval()
            
            # Get yes/no token IDs
            self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")
            
            if self.yes_token_id is None or self.no_token_id is None:
                raise ValueError("Could not find 'yes' or 'no' tokens in tokenizer")
            
            self._predict_fn = self._predict_causal_lm
        else:
            raise ValueError("model_type must be 'sequence_classification' or 'causal_lm'")
    
    def _predict_seq_cls(self, inputs: dict) -> torch.Tensor:
        """Predict using sequence classification model."""
        with torch.no_grad():
            outputs = self.model(**inputs)
            return torch.sigmoid(outputs.logits).squeeze()
    
    def _predict_causal_lm(self, inputs: dict) -> torch.Tensor:
        """Predict using causal language model."""
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_token_logits = outputs.logits[:, -1, :]
            
            yes_logits = last_token_logits[:, self.yes_token_id]
            no_logits = last_token_logits[:, self.no_token_id]
            
            # Compute softmax probabilities for yes/no
            logits = torch.stack([no_logits, yes_logits], dim=1)
            probs = torch.softmax(logits, dim=1)
            
            return probs[:, 1]  # Return "yes" probabilities
    
    def predict(
        self,
        queries: Union[str, List[str]],
        documents: Union[str, List[str]],
        instruction: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Union[float, List[float]]:
        """
        Predict relevance scores for query-document pairs.
        
        Args:
            queries: Single query or list of queries
            documents: Single document or list of documents
            instruction: Task instruction (uses default if None)
            batch_size: Batch size for processing (processes all at once if None)
            
        Returns:
            Single score or list of scores
        """
        # Handle single inputs
        single_input = isinstance(queries, str) and isinstance(documents, str)
        if single_input:
            queries = [queries]
            documents = [documents]
        
        if len(queries) != len(documents):
            raise ValueError("Number of queries must match number of documents")
        
        # Format inputs
        formatted_inputs = format_batch_inputs(instruction, queries, documents)
        
        # Process in batches if specified
        if batch_size is not None and len(formatted_inputs) > batch_size:
            all_scores = []
            for i in range(0, len(formatted_inputs), batch_size):
                batch_inputs = formatted_inputs[i:i + batch_size]
                batch_scores = self._process_batch(batch_inputs)
                all_scores.extend(batch_scores)
            scores = all_scores
        else:
            scores = self._process_batch(formatted_inputs)
        
        # Return single score if single input
        return scores[0] if single_input else scores
    
    def _process_batch(self, formatted_inputs: List[str]) -> List[float]:
        """Process a batch of formatted inputs."""
        # Tokenize
        inputs = self.tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        scores = self._predict_fn(inputs)
        
        # Convert to list
        if scores.dim() == 0:  # Single score
            return [scores.item()]
        else:
            return scores.tolist()
    
    def rank_documents(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
        return_scores: bool = False
    ) -> Union[List[int], Tuple[List[int], List[float]]]:
        """
        Rank documents by relevance to query.
        
        Args:
            query: The search query
            documents: List of documents to rank
            instruction: Task instruction
            return_scores: Whether to return scores along with rankings
            
        Returns:
            List of document indices sorted by relevance (and optionally scores)
        """
        queries = [query] * len(documents)
        scores = self.predict(queries, documents, instruction)
        
        # Create (index, score) pairs and sort by score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        rankings = [idx for idx, _ in indexed_scores]
        
        if return_scores:
            sorted_scores = [score for _, score in indexed_scores]
            return rankings, sorted_scores
        else:
            return rankings


# Convenience functions for backward compatibility
def predict_relevance(
    model_name: str,
    query: str,
    document: str,
    instruction: Optional[str] = None
) -> float:
    """
    Quick function to predict relevance for a single query-document pair.
    
    Args:
        model_name: Name or path of the Qwen3 model
        query: The search query
        document: The document to score
        instruction: Task instruction
        
    Returns:
        Relevance score (0-1)
    """
    wrapper = Qwen3RerankerWrapper(model_name)
    return wrapper.predict(query, document, instruction)


def rank_documents(
    model_name: str,
    query: str,
    documents: List[str],
    instruction: Optional[str] = None
) -> List[Tuple[int, float, str]]:
    """
    Quick function to rank documents by relevance.
    
    Args:
        model_name: Name or path of the Qwen3 model
        query: The search query
        documents: List of documents to rank
        instruction: Task instruction
        
    Returns:
        List of (original_index, score, document) tuples sorted by relevance
    """
    wrapper = Qwen3RerankerWrapper(model_name)
    rankings, scores = wrapper.rank_documents(query, documents, instruction, return_scores=True)
    
    return [(idx, scores[i], documents[idx]) for i, idx in enumerate(rankings)]


# Example usage
if __name__ == "__main__":
    # Example 1: Using the wrapper class
    print("🚀 Testing Qwen3 Reranker Wrapper")
    
    model_name = "Qwen/Qwen3-Reranker-0.6B"  # or your converted model path
    wrapper = Qwen3RerankerWrapper(model_name, model_type="causal_lm")
    
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]
    
    # Get scores
    scores = wrapper.predict([query] * len(documents), documents)
    print(f"Scores: {scores}")
    
    # Rank documents
    rankings = wrapper.rank_documents(query, documents, return_scores=True)
    print(f"Rankings: {rankings}")
    
    # Example 2: Using convenience functions
    print("\n🎯 Testing convenience functions")
    
    single_score = predict_relevance(model_name, query, documents[1])
    print(f"Single score: {single_score}")
    
    ranked_docs = rank_documents(model_name, query, documents)
    for i, (idx, score, doc) in enumerate(ranked_docs):
        print(f"{i+1}. Score: {score:.4f} - {doc[:60]}...")