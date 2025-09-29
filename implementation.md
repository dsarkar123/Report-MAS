# Implementation Guide

This document provides step-by-step instructions for setting up and running the MAS Notice 758 analysis program.

## 1. Prerequisites

- Python 3.8 or higher installed on your system.
- Access to an OpenAI API key.

## 2. Setup

### 2.1. Clone the Repository

First, ensure you have all the project files in a local directory.

### 2.2. Install Dependencies

This project requires several Python packages. You can install them using the provided `requirements.txt` file.

Open your terminal or command prompt, navigate to the project directory, and run the following command:

```bash
pip install -r requirements.txt
```

This will install the following packages:
- `pandas`: For data manipulation and creating Excel files.
- `llama-index`: The core framework for building the RAG application.
- `openai`: To interact with the OpenAI API.
- `openpyxl`: Required by pandas to write `.xlsx` files.

## 3. Configuration

### Set OpenAI API Key

The program requires an OpenAI API key to function. You must set this key as an environment variable.

**For macOS/Linux:**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

**For Windows (Command Prompt):**
```bash
set OPENAI_API_KEY="your_api_key_here"
```

**For Windows (PowerShell):**
```bash
$env:OPENAI_API_KEY="your_api_key_here"
```

Replace `"your_api_key_here"` with your actual OpenAI API key.

## 4. Running the Application

Once the dependencies are installed and the API key is configured, you can run the program.

Execute the `main.py` script from your terminal:

```bash
python main.py
```

### 4.1. Choosing a Query

After starting the application, it will prompt you to choose which query to run.

```
Which query would you like to run? (1 or 2, or 'exit' to quit):
```

- Enter `1` to run the **Amendment Analysis**.
- Enter `2` to run the **Action Points and Business Rules Extraction**.
- Enter `exit` to close the application.

### 4.2. Output Files

The program will generate an Excel report based on the chosen query.

- **Query 1 Output**: `MAS_Notice_758_Amendment_Analysis.xlsx`
- **Query 2 Output**: `MAS_Notice_758_Action_Points_and_Business_Rules.xlsx`

If the program encounters an error while processing the language model's response, it will save the raw output to a text file (`response_query_1.txt` or `response_query_2.txt`) for debugging purposes.

## 5. Understanding the Queries

### Query 1: Amendment Analysis

This query analyzes the MAS Notice 758 document to identify and list all amendments. It provides a structured report comparing the original text with the new text and highlighting the changes.

### Query 2: Action Points and Business Rules

This query extracts all distinct action points and business rules from the document. It classifies each item and provides details such as the responsible department, compliance deadlines, and simplified interpretations of the rules.