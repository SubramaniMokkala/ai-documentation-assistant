
# Document Processor
# Handles parsing and chunking of PDF and text documents


import PyPDF2
import os
from typing import List, Dict


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from PDF file
    
    Args:
        pdf_file: Uploaded PDF file object (from Streamlit)
    
    Returns:
        str: Extracted text
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        return text
    
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")


def extract_text_from_txt(txt_file) -> str:
    """
    Extract text from TXT file
    
    Args:
        txt_file: Uploaded text file object
    
    Returns:
        str: File contents
    """
    try:
        # Read as string
        text = txt_file.read().decode('utf-8')
        return text
    
    except Exception as e:
        raise Exception(f"Error reading text file: {str(e)}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Split text into overlapping chunks for better context preservation
    
    Args:
        text: Full text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of dictionaries with chunk text and metadata
    
    Why chunking?
    - AI models have token limits
    - Smaller chunks = more precise retrieval
    - Overlap preserves context across chunks
    """
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        # Get chunk
        end = start + chunk_size
        chunk_text = text[start:end]
        
        # Skip very small chunks at the end
        if len(chunk_text.strip()) < 50:
            break
        
        # Create chunk metadata
        chunk = {
            'id': chunk_id,
            'text': chunk_text.strip(),
            'start_char': start,
            'end_char': end,
            'length': len(chunk_text.strip())
        }
        
        chunks.append(chunk)
        
        # Move start position (with overlap)
        start = end - overlap
        chunk_id += 1
    
    return chunks


def process_document(file, filename: str) -> Dict:
    """
    Process uploaded document and return chunks with metadata
    
    Args:
        file: Uploaded file object
        filename: Name of the file
    
    Returns:
        Dictionary with document info and chunks
    """
    # Determine file type
    file_extension = filename.lower().split('.')[-1]
    
    # Extract text based on file type
    if file_extension == 'pdf':
        text = extract_text_from_pdf(file)
    elif file_extension in ['txt', 'md']:
        text = extract_text_from_txt(file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    # Chunk the text
    chunks = chunk_text(text)
    
    # Create document metadata
    document = {
        'filename': filename,
        'file_type': file_extension,
        'total_chars': len(text),
        'num_chunks': len(chunks),
        'chunks': chunks
    }
    
    return document


def get_document_stats(document: Dict) -> Dict:
    """
    Get statistics about a processed document
    
    Args:
        document: Processed document dictionary
    
    Returns:
        Dictionary of statistics
    """
    stats = {
        'filename': document['filename'],
        'file_type': document['file_type'],
        'total_characters': document['total_chars'],
        'number_of_chunks': document['num_chunks'],
        'avg_chunk_size': document['total_chars'] / document['num_chunks'] if document['num_chunks'] > 0 else 0
    }
    
    return stats


# Test the module
if __name__ == "__main__":
    print("Testing Document Processor Module...")
    
    # Create sample text
    sample_text = """
    This is a sample document for testing the chunking functionality.
    The document processor will split this text into smaller chunks
    with overlapping content to preserve context between chunks.
    This is important for maintaining coherence when the AI retrieves
    and processes the information.
    """ * 10  # Repeat to make it longer
    
    print(f"\nSample text length: {len(sample_text)} characters")
    
    # Test chunking
    chunks = chunk_text(sample_text, chunk_size=200, overlap=50)
    
    print(f"Number of chunks created: {len(chunks)}")
    print(f"\nFirst chunk preview:")
    print(chunks[0]['text'][:100] + "...")
    
    print("\nDocument processor module working correctly!")