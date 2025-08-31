import os
import re
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from typing import List, Dict
import csv

_ = load_dotenv(find_dotenv())
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

system_content = """
### CONTEXT ###
You are Phd professor working in a literature review for your research, you need to read the abstract, introduction, and conclusion of the article below. 

- Area of Activity: Identify the primary business function or field the article addresses. Is it marketing, operations, customer service, fraud detection, human resources, etc.? Look for keywords related to  areas.

- Context: Determine the specific problem, technology, or scenario the article discusses. What is the article's core topic? For example, if the area of activity is marketing, the context could be "using AI for customer segmentation" or "predictive analytics in ad campaigns." The context is the "what" of the article.

- AI Implementation State: Analyze how AI is being used. Match the article's AI application to one of the four maturity stages: Experimentation and Proof of Concept: The article talks about a pilot project, a test, or an early-stage exploration. The language will be speculative, focusing on potential rather than widespread use;  Tactical and Departmental Adoption: The article describes AI being used to solve a specific problem within a single department or function. It's a practical application, but not yet a company-wide strategy; Corporate Strategy and Optimization: The article discusses a company-wide initiative or the integration of AI across multiple departments. It frames AI as a strategic tool for business optimization; AI-Driven Transformation and Innovation: The article focuses on how AI is creating new business models, products, or services. It's about fundamental change, not just efficiency. Look for terms like "redefining the industry" or "creating new value propositions."


Article:
“{full_text}”

### AUDIENCE ###
The audience is are another academic professionals.

### STYLE & TONE ###
Your style should be clear, concise, and professional. The tone should be informative and confident.

### OBJECTIVE & RESPONSE FORMAT ###
You need to determine Area of Activity, Context, and AI Implementation State the article, the answer must be short and directly. 
"""


class UserFormat(BaseModel):
    area_of_activity: str
    context: str
    ai_implementation_state: str

def list_pdf_files(directory: str) -> list:
    """
    Lists all files ending with the .pdf extension within a given directory.

    Args:
        directory (str): The path to the directory to search.

    Returns:
        list: A list of full file paths for all found PDF files.
    """
    pdf_files = []

    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return pdf_files

    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            # Construct the full path to the file
            full_path = os.path.join(directory, filename)
            pdf_files.append(full_path)

    print(f"Found {len(pdf_files)} PDF file(s).")
    return pdf_files

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Reads the content of a PDF file and returns it as a single text string.

    Args:
        pdf_path (str): The full path to the PDF file.

    Returns:
        str: The extracted text content of the PDF. Returns an empty string on failure.
    """
    print(f"\n--- Extracting text from: {os.path.basename(pdf_path)} ---")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        print("Text extraction successful.")
        return text
    except Exception as e:
        print(f"Could not read PDF file '{pdf_path}'. Error: {e}")
        return ""

def get_completion_from_messages(messages: List[Dict[str, str]],
                                 model="gpt-4.1-mini",
                                 temperature=0) -> str:
    """
    Generates a response from an OpenAI language model based on a conversation history.

    Args:
        messages (List[Dict[str, str]]):
            A list of dictionaries representing the conversation history.
            Each dictionary should have two keys:
                - 'role' (str): The role of the speaker in the conversation, such as "user", "assistant", or "system".
                - 'content' (str): The content of the message for that role.
        model (str, optional):
            The name of the model to use for generating the response.
            Defaults to "gpt-5-mini".
        temperature (float, optional):
            The sampling temperature to use for response generation.
            Higher values (e.g., 0.8) produce more creative and diverse outputs, while lower values (e.g., 0.2) produce more deterministic results.
            Defaults to 0.

    Returns:
        str: The content of the model's response to the conversation history provided.

    Example:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        response = get_completion_from_messages(messages)
        print(response)  # Output: "The capital of France is Paris."
    """
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format=UserFormat
    )
    return completion.choices[0].message.parsed

def save_csv(data: List[Dict[str, str]], filename: str) -> None:
    """
    Saves the content of a list of dictionaries into a CSV file.

    Args:
        data (List[Dict[str, str]]):
            A list of dictionaries where each dictionary represents a row of data.
            The keys of the dictionaries will be used as the column headers.
        filename (str):
            The name of the CSV file to save the data to.

    Returns:
        None

    Example:
        data = [
            {"name": "Alice", "age": "30", "city": "New York"},
            {"name": "Bob", "age": "25", "city": "Los Angeles"}
        ]
        save_dicts_to_csv(data, "output.csv")
    """
    if not data:
        raise ValueError("The data list is empty. Cannot save to a CSV file.")

    # Extract headers from the keys of the first dictionary
    headers = data[0].keys()

    # Write data to CSV
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

    print(f"Wrote to {filename}")

def main():

    # Path directory
    pdf_directory = 'articles'

    # Read pdf files
    pdf_files = list_pdf_files(pdf_directory)

    # Set model
    model = "gpt-4.1-mini"

    # Instantiate the list
    articles = []

    # Loop through each found PDF file
    for index, pdf_path in enumerate(pdf_files):

        # Instantiate dictionary
        article = {}

        # Set article name
        _, article['name'], _ = re.split(r'\/|\.pdf', pdf_path)

        # Extract the full text from the PDF
        full_text = extract_text_from_pdf(pdf_path)

        if not full_text:
            print(f"Skipping file {os.path.basename(pdf_path)} due to extraction error.")
            continue

        user = get_completion_from_messages(
            messages=[{"role": "system", "content": system_content},
                      {"role": "user", "content": "Article: {}".format(full_text)}],
            model=model
        )

        articles.append({**article, **user.model_dump(mode='json')})

        # Print article index
        print(f'Index {index}, Article: {article["name"]}')

    # Save file to articles to .csv
    save_csv(data=articles, filename=f'data/data_extraction_{model}.csv')


if __name__ == "__main__":
    main()
