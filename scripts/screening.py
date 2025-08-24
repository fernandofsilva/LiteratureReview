from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from openai import OpenAI
import os
import csv


_ = load_dotenv(find_dotenv())
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

system_content = """
### CONTEXT ###
You are Phd professor working in a literature review for your research, you need to analyze an article's abstract and determine whether it matches your research topic. The research aims to investigate how small and medium-sized enterprises (SMEs) are using artificial intelligence (AI), machine-learning (ML), Deep Learning (DL), Artificial Neural Networks (ANN) or Neural Network (NN) in their operations. The research seeks to identify the benefits, challenges, and best practices related to the adoption and use of this technologies, with the goal of understanding the influence of this technology on organizational performance and company competitiveness.

Abstract:
“{article_abstract}”

### OBJECTIVE & RESPONSE FORMAT ###
Your objective is to determine whether an article is related to your research and answer with the reason it should be part of your literature review, the reason must be short and directly. 
"""

class UserFormat(BaseModel):
    match: bool
    reason: str

def read_csv(file_path: str) -> List[Dict[str, str]]:
    """
    Reads a CSV file and stores its content in a list of dictionaries.

    Each row in the CSV file is converted into a dictionary where the keys are the column headers
    and the values are the corresponding row values. The function returns a list of these dictionaries.

    Args:
        file_path (str): The path to the CSV file to be read.

    Returns:
        List[Dict[str, str]]: A list of dictionaries representing the rows of the CSV file.
    """
    articles: List[Dict[str, str]] = []

    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            articles.append(row)

    return articles

def parse_notes_field(articles: List[Dict[str, str]]) -> None:
    """
    Processes a list of articles and standardizes the 'notes' field for each article.

    Parameters:
        articles (List[Dict[str, str]]): A list of dictionaries where each dictionary represents
                                         an article with a 'notes' field.

    Behavior:
        - If the 'notes' field contains 'Included', it is updated to 'Included'.
        - If the 'notes' field contains 'Excluded', it is updated to 'Excluded'.
        - If the 'notes' field contains 'Maybe', it is updated to 'Maybe'.
        - If none of the above are found, the 'notes' field is set to ''.

    Returns:
        None
    """

    for article in articles:

        match article['notes']:
            case string if 'Included' in article['notes']:
                article['notes'] = 'Included'
            case string if 'Excluded' in article['notes']:
                article['notes'] = 'Excluded'
            case string if 'Maybe' in article['notes']:
                article['notes'] = 'Maybe'
            case _:
                article['notes'] = ''

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


def main():

    # Read articles
    articles = read_csv('data/articles.csv')

    # Parte Nodes field
    parse_notes_field(articles)

    # Set model
    model = "gpt-4.1-mini"

    # Loop over the articles
    for index, article in enumerate(articles):

        # Create new criteria field in the dict
        user = get_completion_from_messages(
            messages=[{"role": "system", "content": system_content},
                      {"role": "user", "content": "Abstract: {}".format(article['abstract'])}],
            model=model
        )

        articles[index] = {**article, **user.model_dump(mode='json')}

        print(f"Article number {index}, percentage conclusion {index/len(articles)*100:.2f}%")

    # Save file to articles to .csv
    save_csv(data=articles, filename=f'data/articles_{model}.csv')


if __name__ == "__main__":
    main()
