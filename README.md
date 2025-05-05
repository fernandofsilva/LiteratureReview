# Literature Review

## Project Overview

This project supports the master's dissertation titled:

**“Inteligência Artificial como Motor de Competitividade em Pequenas e Médias Empresas: Automatização, Personalização e Tomada de Decisão”**  
by **Fernando Ferreira da Silva**, under the guidance of **Prof. Adrian Kemmer Cernev**  
(FGV EAESP – Line of Research: Operations and Innovation)

The dissertation investigates how small and medium enterprises (SMEs) are adopting artificial intelligence (AI) to enhance their operational efficiency and competitiveness. To assist in this research, this project automates the analysis and categorization of academic abstracts related to AI adoption using OpenAI’s language models.

## Purpose

The system is designed to:
- Simulate academic feedback from a PhD-level professor on AI-related abstracts.
- Automatically classify academic abstracts into relevant categories (e.g., AI, Economics, Public Health).
- Support the early stages of literature review through intelligent filtering and organization of sources.

## Features

- **AI-Powered Abstract Review**: Sends research abstracts to OpenAI's API and retrieves feedback framed as a PhD professor.
- **Abstract Classification**: Categorizes academic abstracts into pre-defined domains using few-shot learning.
- **File-Based Input**: Reads abstracts from `.txt` files for streamlined processing.
- **Test Mode**: Includes usage examples and test summaries to verify functionality.

## Technologies

- Python 3.12+
- [OpenAI API](https://platform.openai.com/)
- `pydantic`, `dotenv`
- GPT-4 / GPT-4.1-mini model
- CLI / file-based interface

## How It Works

1. Load an abstract from a `.txt` file.
2. Send the abstract to OpenAI’s API:
   - Get **professor-style feedback** via `ask_phd_professor()`
   - Or get a **category classification** via `classificar_resumo()`
3. Print or log the structured output for later inclusion in the dissertation.

## Getting Started

### Installation

```bash
git clone https://github.com/yourusername/LiteratureReview.git
cd LiteratureReview
pip install -r requirements.txt
```

### Configuration

Create a `.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY=your-openai-key-here
```

### Usage

To get professor feedback on a research abstract:

```bash
python model_test.py
```

Make sure you have a file named `abstract.txt` with the content to be analyzed.

## Research Background

The dissertation aims to answer:

> **"How are small and medium enterprises using artificial intelligence?"**

### Methodology Summary:
- Multiple case studies of service-sector SMEs
- Semi-structured interviews with decision-makers
- Qualitative data analysis using Gioia methodology and Pozzebon & Diniz's multilevel framework

## Contributing

This project is academic in nature and not open for external contributions, but feel free to adapt or extend it for your own research.

## License

MIT License
