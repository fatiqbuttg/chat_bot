# AI Question Answering Bot

## Overview

The AI Question Answering Bot is a Python-based application designed to answer questions based on the content of a PDF document. It utilizes the LangChain framework and Google Generative AI for question-answering tasks. The bot can process a PDF file, split its content into manageable chunks, and then use advanced language models to provide accurate answers to user queries.

## Features

- **PDF Processing**: Load and process PDF documents to extract text.
- **Text Splitting**: Divide text into chunks for better handling and analysis.
- **Question Answering**: Answer questions based on the content of the PDF using Google Generative AI.
- **User Interaction**: Interactive command-line interface for asking questions and quitting the program.

## Requirements

To run the AI Question Answering Bot, you need to have Python installed on your machine. The project relies on the following Python libraries:

- `langchain_community`
- `langchain_google_genai`
- `langchain`
- `langchain_chroma`
- `langchain_text_splitters`
- `langchain_core`

## Setup Instructions

Follow these steps to set up and run the bot:

1. **Clone the Repository**

2. **Navigate to the Project Directory:**
      - `cd chat_bot`
3. **Create a virtual environment to manage dependencies:**
      - `python3 -m venv code_env`
   
4. **Install Dependencies:**
      - `pip3 install -r requirements.txt`

5. **Add your api key in code**
6. **Run the Script:**
      - `python3 bot.py`

