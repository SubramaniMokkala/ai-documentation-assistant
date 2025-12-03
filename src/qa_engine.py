# Question Answering Engine
# Uses local Hugging Face models for document Q&A

from transformers import pipeline
from typing import List, Dict


class QAEngine:
    """
    Question answering engine using local transformer models
    
    How it works (RAG - Retrieval Augmented Generation):
    1. User asks a question
    2. Embeddings module finds relevant document chunks (retrieval)
    3. This module generates an answer from those chunks (generation)
    """
    
    def __init__(self):
        """
        Initialize the QA model
        
        We use 'deepset/roberta-base-squad2':
        - Trained on SQuAD 2.0 dataset (100k+ Q&A pairs)
        - Good at extractive Q&A (finding answers in text)
        - Runs locally, no API needed
        - ~500MB model
        """
        print("Loading QA model (this may take a minute on first run)...")
        
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=-1  # Use CPU (set to 0 for GPU if available)
            )
            print("QA model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to simple extraction method...")
            self.qa_pipeline = None
    
    
    def answer_question(self, question: str, contexts: List[Dict], 
                       max_context_length: int = 2000) -> Dict:
        """
        Answer a question using retrieved contexts
        
        Args:
            question: User's question
            contexts: List of relevant chunks from embeddings search
            max_context_length: Max characters to send to model
        
        Returns:
            Dictionary with answer, confidence, and sources
        """
        if not contexts or len(contexts) == 0:
            return {
                'answer': "I couldn't find relevant information in the documents to answer this question.",
                'confidence': 0.0,
                'sources': []
            }
        
        # Combine top contexts
        combined_context = self._combine_contexts(contexts, max_context_length)
        
        # If model failed to load, use simple extraction
        if self.qa_pipeline is None:
            return self._simple_answer(question, contexts)
        
        try:
            # Use transformer model for Q&A
            result = self.qa_pipeline(
                question=question,
                context=combined_context
            )
            
            # Extract source documents
            sources = self._extract_sources(contexts[:3])  # Top 3 sources
            
            return {
                'answer': result['answer'],
                'confidence': result['score'],
                'sources': sources,
                'context_used': combined_context[:200] + "..."  # Preview
            }
        
        except Exception as e:
            print(f"Error during Q&A: {e}")
            return self._simple_answer(question, contexts)
    
    
    def _combine_contexts(self, contexts: List[Dict], max_length: int) -> str:
        """
        Combine multiple context chunks into one string
        
        Args:
            contexts: List of context chunks
            max_length: Maximum total length
        
        Returns:
            Combined context string
        """
        combined = ""
        
        for ctx in contexts:
            chunk_text = ctx['text']
            
            # Check if adding this chunk exceeds max length
            if len(combined) + len(chunk_text) > max_length:
                break
            
            combined += chunk_text + "\n\n"
        
        return combined.strip()
    
    
    def _extract_sources(self, contexts: List[Dict]) -> List[Dict]:
        """
        Extract source information from contexts
        
        Args:
            contexts: List of context chunks
        
        Returns:
            List of source dictionaries
        """
        sources = []
        seen_docs = set()
        
        for ctx in contexts:
            doc_name = ctx.get('source_document', 'Unknown')
            
            # Avoid duplicate documents
            if doc_name not in seen_docs:
                sources.append({
                    'document': doc_name,
                    'similarity': ctx.get('similarity_score', 0),
                    'preview': ctx['text'][:150] + "..."
                })
                seen_docs.add(doc_name)
        
        return sources
    
    
    def _simple_answer(self, question: str, contexts: List[Dict]) -> Dict:
        """
        Fallback method: return most relevant context as answer
        
        Args:
            question: User's question
            contexts: Retrieved contexts
        
        Returns:
            Answer dictionary
        """
        if not contexts:
            return {
                'answer': "No relevant information found.",
                'confidence': 0.0,
                'sources': []
            }
        
        # Return top context
        top_context = contexts[0]
        sources = self._extract_sources(contexts[:3])
        
        return {
            'answer': top_context['text'],
            'confidence': top_context.get('similarity_score', 0.5),
            'sources': sources,
            'note': 'Using simple extraction (full model not available)'
        }
    
    
    def generate_summary(self, contexts: List[Dict], max_contexts: int = 3) -> str:
        """
        Generate a summary of multiple contexts
        
        Args:
            contexts: List of context chunks
            max_contexts: Maximum number of contexts to summarize
        
        Returns:
            Summary string
        """
        if not contexts:
            return "No information available."
        
        # Take top contexts
        top_contexts = contexts[:max_contexts]
        
        summary_parts = []
        for i, ctx in enumerate(top_contexts, 1):
            doc_name = ctx.get('source_document', 'Unknown')
            text = ctx['text'][:200] + "..." if len(ctx['text']) > 200 else ctx['text']
            
            summary_parts.append(f"**From {doc_name}:**\n{text}")
        
        return "\n\n".join(summary_parts)


# Test the module
if __name__ == "__main__":
    print("Testing QA Engine Module...")
    print("=" * 60)
    
    # Initialize engine
    engine = QAEngine()
    
    # Sample contexts (as if retrieved from embeddings)
    sample_contexts = [
        {
            'text': 'Python is a high-level programming language widely used for data science, web development, and automation. It is known for its simple syntax and large ecosystem of libraries.',
            'source_document': 'python_intro.txt',
            'similarity_score': 0.89
        },
        {
            'text': 'Data science involves extracting insights from data using statistical methods, machine learning, and visualization tools.',
            'source_document': 'data_science_guide.pdf',
            'similarity_score': 0.72
        }
    ]
    
    # Test question answering
    print("\nTesting question answering...")
    question = "What is Python used for?"
    
    result = engine.answer_question(question, sample_contexts)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nSources:")
    for source in result['sources']:
        print(f"  - {source['document']} (similarity: {source['similarity']:.2f})")
    
    print("\n" + "=" * 60)
    print("QA Engine working correctly")
    print("\nKey features:")
    print("  - Extractive Q&A from documents")
    print("  - Source attribution")
    print("  - Confidence scoring")
    print("  - Runs 100% locally (no API calls)")