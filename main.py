import openai
import os

# Set up the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or replace with your actual API key


def load_document(file_path):
    """Load document from file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def generate_question_answer_pairs(text, num_pairs=5):
    """
    Generate question-answer pairs from the provided text using GPT.

    Args:
        text (str): The content of the document.
        num_pairs (int): Number of question-answer pairs to generate.

    Returns:
        list of tuples: A list of (question, answer) pairs.
    """
    prompt = f"""
    You are a helpful assistant. Below is a document. Based on the document, generate {num_pairs} question-answer pairs.

    Document:
    {text}

    The questions should be diverse and cover different aspects of the document. Provide the questions and answers in the following format:
    1. Question: <question> Answer: <answer>
    2. Question: <question> Answer: <answer>
    """

    response = openai.Completion.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        prompt=prompt,
        max_tokens=500,
        temperature=0.7,
        n=1,
        stop=None
    )

    # Parse the response
    qa_text = response.choices[0].text.strip()
    qa_pairs = []
    for qa in qa_text.split("\n"):
        # Parse each question-answer pair from the output format
        if "Question:" in qa and "Answer:" in qa:
            question_answer = qa.split("Answer:")
            question = question_answer[0].replace("Question:", "").strip()
            answer = question_answer[1].strip()
            qa_pairs.append((question, answer))

    return qa_pairs


def display_qa_pairs(qa_pairs):
    """Display the generated question-answer pairs."""
    for idx, (question, answer) in enumerate(qa_pairs, 1):
        print(f"{idx}. Question: {question}\n   Answer: {answer}\n")


# Main function to run the program
def main():
    # Load the document
    file_path = "document.txt"  # Replace with your document path
    document_text = load_document(file_path)

    # Generate question-answer pairs
    qa_pairs = generate_question_answer_pairs(document_text, num_pairs=5)

    # Display the question-answer pairs
    display_qa_pairs(qa_pairs)


if __name__ == "__main__":
    main()
