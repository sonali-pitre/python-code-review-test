import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or use your key directly: "your-api-key"


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


def display_summary(summary):
    print("\n📄 Document Summary:\n" + "-" * 60)
    print(summary)


def display_qa_pairs(qa_pairs):
    print("\n❓ Question-Answer Pairs:\n" + "-"*60)
    for i, (q, a) in enumerate(qa_pairs, 1):
        print(f"{i}. Q: {q}\n   A: {a}\n")

def main():
    # Replace this with your actual document path
    file_path = "document.txt"

    # Step 1: Load the document
    document_text = load_document(file_path)

    # Step 2: Summarize the document
    summary = summarize_document(document_text)
    display_summary(summary)

    # Step 3: Generate Q&A pairs
    qa_pairs = generate_question_answer_pairs(document_text, num_pairs=5)
    display_qa_pairs(qa_pairs)

if __name__ == "__main__":
    main()

