from typing import List, Dict, Any
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from openai import OpenAI
import os
import csv

_ = load_dotenv(find_dotenv())
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


def ask_phd_professor(
    abstract: str,
    model: str = "gpt-4.1-mini") -> str:
    """
    Send a student’s abstract to the OpenAI API with instructions to respond as a PhD professor.

    Args:
        abstract (str): The student’s abstract describing AI adoption, implementation, or use.
        model (str): The OpenAI model to use.

    Returns:
        str: The professor’s detailed feedback or response.
    """

    # Define system prompt instructing the model to act as a PhD professor
    system_prompt = (
        "You are a highly knowledgeable PhD professor in AI research. "
        "Provide detailed, constructive, and scholarly feedback on the student's abstract, "
        "focusing on AI adoption, implementation, or use."
    )

    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": abstract}
    ]

    # Make the API request
    response = client.responses.create(
        model=model,
        input=messages,
        temperature=0
    )

    # Extract and return the professor's reply
    return response.output_text

def read_abstract_from_file(file_path: str) -> str:
    """
    Read the abstract from a specified text file.

    Args:
        file_path (str): Path to the file containing the abstract.

    Returns:
        str: Contents of the file as a string.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().strip()
    print(f"[DEBUG] Abstract read from file '{file_path}': {content}")
    return content


if __name__ == "__main__":
    # Example usage: replace the abstract string below with the student's abstract.
    example_abstract = read_abstract_from_file("abstract.txt")
    feedback = ask_phd_professor(example_abstract)
    print("Professor's Feedback:\n", feedback)

import openai

# Substitua com sua chave da API
openai.api_key = "sua-chave-aqui"

# Exemplo de mensagens iniciais: system + exemplos (few-shot)
messages = [
    {
        "role": "system",
        "content": "Você é um assistente especializado em classificar resumos de artigos acadêmicos. Sua tarefa é analisar cada resumo fornecido e atribuí-lo a uma das categorias pré-definidas com base em exemplos."
    },
    {
        "role": "user",
        "content": """Exemplos de classificação:

Resumo: Este artigo apresenta uma nova abordagem para o treinamento de redes neurais convolucionais aplicadas à visão computacional.
Classificação: Inteligência Artificial

Resumo: O estudo investiga o impacto de políticas públicas na distribuição de renda no Brasil entre 2000 e 2020.
Classificação: Economia

Resumo: Um levantamento dos principais métodos de prevenção de doenças cardiovasculares em populações urbanas.
Classificação: Saúde Pública"""
    }
]


# Função para classificar um novo resumo
def classificar_resumo(resumo):
    # Adiciona o resumo a ser classificado à conversa
    messages.append({
        "role": "user",
        "content": f"Resumo: {resumo}\nQual é a classificação?"
    })

    # Faz a chamada à API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )

    # Extrai a resposta
    resposta = response['choices'][0]['message']['content']

    # Remove o último input do usuário para manter o contexto limpo
    messages.pop()

    return resposta


# Teste
resumo_novo = "Este trabalho propõe um novo modelo para estimar o risco de crédito utilizando aprendizado de máquina supervisionado."
resultado = classificar_resumo(resumo_novo)
print("Resposta do modelo:", resultado)