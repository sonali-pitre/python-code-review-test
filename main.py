import openai
import os
import nltk
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import spacy
from textstat import textstat
import hashlib
import json
import concurrent.futures
from functools import lru_cache
from pathlib import Path
import time

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def load_document(file_path):
    """Load document from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def summarize_document(text):
    """
    Summarize the given document using GPT.

    Args:
        text (str): The document text.

    Returns:
        str: Summary of the document.
    """
    prompt = f"""
    Summarize the following document in a concise paragraph that captures the main ideas:

    Document:
    {text}

    Summary:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300
    )

    return response['choices'][0]['message']['content'].strip()


def generate_question_answer_pairs(text, num_pairs=5):
    """
    Generate question-answer pairs from the document.

    Args:
        text (str): The document text.
        num_pairs (int): Number of Q&A pairs to generate.

    Returns:
        list of (str, str): List of (question, answer) tuples.
    """
    prompt = f"""
    You are a helpful assistant. Based on the document below, generate {num_pairs} diverse question-answer pairs.

    Document:
    {text}

    Provide the output in this format:
    1. Question: ... Answer: ...
    2. Question: ... Answer: ...
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=700
    )

    qa_text = response['choices'][0]['message']['content'].strip()

    # Parse the output into a list of (question, answer) pairs
    qa_pairs = []
    for line in qa_text.split("\n"):
        if "Question:" in line and "Answer:" in line:
            q_part, a_part = line.split("Answer:")
            question = q_part.split("Question:")[-1].strip()
            answer = a_part.strip()
            qa_pairs.append((question, answer))
    return qa_pairs


def extract_key_topics(text, num_topics=5):
    """
    Extract key topics from the document using word frequency analysis.
    
    Args:
        text (str): The document text
        num_topics (int): Number of top topics to extract
        
    Returns:
        list: Top topics with their frequencies
    """
    # Tokenize and clean text
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Get top topics
    return word_freq.most_common(num_topics)


def analyze_sentiment(text):
    """
    Analyze the sentiment of the document.
    
    Args:
        text (str): The document text
        
    Returns:
        dict: Sentiment analysis results
    """
    blob = TextBlob(text)
    
    # Get overall sentiment
    sentiment = blob.sentiment
    
    # Analyze sentiment per sentence
    sentences = sent_tokenize(text)
    sentence_sentiments = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
    
    return {
        'overall_polarity': sentiment.polarity,
        'overall_subjectivity': sentiment.subjectivity,
        'sentence_polarities': sentence_sentiments,
        'average_sentence_polarity': sum(sentence_sentiments) / len(sentence_sentiments)
    }


def extract_entities(text):
    """
    Extract named entities from the document using spaCy.
    
    Args:
        text (str): The document text
        
    Returns:
        dict: Dictionary of entities grouped by type
    """
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    
    return entities


def calculate_readability(text):
    """
    Calculate various readability metrics for the document.
    
    Args:
        text (str): The document text
        
    Returns:
        dict: Dictionary of readability metrics
    """
    return {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text),
        'smog_index': textstat.smog_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
    }


def display_summary(summary):
    print("\n📄 Document Summary:\n" + "-" * 60)
    print(summary)


def display_qa_pairs(qa_pairs):
    print("\n❓ Question-Answer Pairs:\n" + "-"*60)
    for i, (q, a) in enumerate(qa_pairs, 1):
        print(f"{i}. Q: {q}\n   A: {a}\n")


def display_enhanced_analysis(topics, sentiment, entities, readability):
    """Display the results of enhanced analysis."""
    print("\n📊 Enhanced Analysis Results:\n" + "-"*60)
    
    print("\n📌 Key Topics:")
    for topic, freq in topics:
        print(f"- {topic}: {freq} occurrences")
    
    print("\n😊 Sentiment Analysis:")
    print(f"- Overall Polarity: {sentiment['overall_polarity']:.2f} (-1 negative to 1 positive)")
    print(f"- Overall Subjectivity: {sentiment['overall_subjectivity']:.2f} (0 objective to 1 subjective)")
    print(f"- Average Sentence Polarity: {sentiment['average_sentence_polarity']:.2f}")
    
    print("\n🏷️ Named Entities:")
    for entity_type, entity_list in entities.items():
        print(f"- {entity_type}: {', '.join(set(entity_list))}")
    
    print("\n📚 Readability Metrics:")
    print(f"- Flesch Reading Ease: {readability['flesch_reading_ease']:.1f}")
    print(f"- Flesch-Kincaid Grade Level: {readability['flesch_kincaid_grade']:.1f}")
    print(f"- Gunning Fog Index: {readability['gunning_fog']:.1f}")
    print(f"- SMOG Index: {readability['smog_index']:.1f}")
    print(f"- Automated Readability Index: {readability['automated_readability_index']:.1f}")


class DocumentProcessor:
    def __init__(self, cache_dir="cache"):
        """Initialize the document processor with caching support."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.chunk_size = 2000  # tokens per chunk
        self.max_concurrent_files = 3

    def _get_cache_path(self, content, operation):
        """Generate a cache file path based on content hash and operation."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return self.cache_dir / f"{operation}_{content_hash}.json"

    def _cache_result(self, content, operation, result):
        """Cache the result of an operation."""
        cache_path = self._get_cache_path(content, operation)
        with cache_path.open('w', encoding='utf-8') as f:
            json.dump(result, f)

    def _get_cached_result(self, content, operation):
        """Retrieve cached result if available."""
        cache_path = self._get_cache_path(content, operation)
        if cache_path.exists():
            with cache_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def chunk_document(self, text):
        """Split document into smaller chunks for processing."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_tokens = len(word_tokenize(sentence))
            if current_size + sentence_tokens > self.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_size += sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    @lru_cache(maxsize=100)
    def process_chunk(self, chunk_text):
        """Process a single chunk of text with caching."""
        return {
            'summary': summarize_document(chunk_text),
            'qa_pairs': generate_question_answer_pairs(chunk_text, num_pairs=2),
            'topics': extract_key_topics(chunk_text),
            'sentiment': analyze_sentiment(chunk_text),
            'entities': extract_entities(chunk_text),
            'readability': calculate_readability(chunk_text)
        }

    def process_documents(self, file_paths):
        """Process multiple documents in parallel."""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_files) as executor:
            future_to_path = {
                executor.submit(self.process_document, path): path 
                for path in file_paths
            }
            
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")
        
        return results

    def process_document(self, file_path):
        """Process a single document with chunking and caching."""
        start_time = time.time()
        text = load_document(file_path)
        
        # Check cache first
        cached_result = self._get_cached_result(text, "full_analysis")
        if cached_result:
            print(f"Using cached results for {file_path}")
            return cached_result
        
        # Process in chunks
        chunks = self.chunk_document(text)
        chunk_results = []
        
        for chunk in chunks:
            chunk_result = self.process_chunk(chunk)
            chunk_results.append(chunk_result)
        
        # Combine chunk results
        combined_result = self._combine_chunk_results(chunk_results)
        
        # Cache the final result
        self._cache_result(text, "full_analysis", combined_result)
        
        processing_time = time.time() - start_time
        print(f"Processed {file_path} in {processing_time:.2f} seconds")
        
        return combined_result

    def _combine_chunk_results(self, chunk_results):
        """Combine results from multiple chunks into a single result."""
        combined = {
            'summary': '',
            'qa_pairs': [],
            'topics': Counter(),
            'sentiment': {
                'overall_polarity': 0,
                'overall_subjectivity': 0,
                'sentence_polarities': [],
                'average_sentence_polarity': 0
            },
            'entities': {},
            'readability': {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'gunning_fog': 0,
                'smog_index': 0,
                'automated_readability_index': 0
            }
        }
        
        # Combine summaries
        combined['summary'] = ' '.join(r['summary'] for r in chunk_results)
        
        # Combine QA pairs
        for result in chunk_results:
            combined['qa_pairs'].extend(result['qa_pairs'])
        
        # Combine topics
        for result in chunk_results:
            for topic, freq in result['topics']:
                combined['topics'][topic] += freq
        combined['topics'] = combined['topics'].most_common(10)
        
        # Average sentiment metrics
        n_chunks = len(chunk_results)
        for result in chunk_results:
            combined['sentiment']['overall_polarity'] += result['sentiment']['overall_polarity']
            combined['sentiment']['overall_subjectivity'] += result['sentiment']['overall_subjectivity']
            combined['sentiment']['sentence_polarities'].extend(result['sentiment']['sentence_polarities'])
        
        combined['sentiment']['overall_polarity'] /= n_chunks
        combined['sentiment']['overall_subjectivity'] /= n_chunks
        combined['sentiment']['average_sentence_polarity'] = (
            sum(combined['sentiment']['sentence_polarities']) / 
            len(combined['sentiment']['sentence_polarities'])
        )
        
        # Combine entities
        for result in chunk_results:
            for entity_type, entities in result['entities'].items():
                if entity_type not in combined['entities']:
                    combined['entities'][entity_type] = []
                combined['entities'][entity_type].extend(entities)
        
        # Average readability metrics
        for metric in combined['readability']:
            combined['readability'][metric] = sum(
                r['readability'][metric] for r in chunk_results
            ) / n_chunks
        
        return combined


def main():
    # Initialize document processor
    processor = DocumentProcessor(cache_dir="document_cache")
    
    # Single file processing
    file_path = "document.txt"
    result = processor.process_document(file_path)
    
    # Display results
    display_summary(result['summary'])
    display_qa_pairs(result['qa_pairs'])
    display_enhanced_analysis(
        result['topics'],
        result['sentiment'],
        result['entities'],
        result['readability']
    )
    
    # Example of batch processing multiple files
    # file_paths = ["doc1.txt", "doc2.txt", "doc3.txt"]
    # results = processor.process_documents(file_paths)
    # for path, result in results.items():
    #     print(f"\nResults for {path}:")
    #     display_summary(result['summary'])
    #     display_qa_pairs(result['qa_pairs'])
    #     display_enhanced_analysis(
    #         result['topics'],
    #         result['sentiment'],
    #         result['entities'],
    #         result['readability']
    #     )


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")  # Or use your key directly: "your-api-key"
    main()
