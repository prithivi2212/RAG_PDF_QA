RAG PDF Question Answering System

This project is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions based on the contents of a PDF document. Instead of relying on the model to remember information, the system retrieves the most relevant text sections from the PDF and uses them to generate an accurate and grounded answer.

The application works in two modes:

Using the Groq API (cloud-based model).

Using a local language model through Ollama for offline usage.

Both modes can be switched by changing one line of code in the script.

Purpose

The goal of this project is to demonstrate how document-based question answering can be built using embeddings, similarity search, and Large Language Models (LLMs). This is useful for use cases such as legal documents, reports, research papers, manuals, textbooks and internal company documents.

How It Works

The script extracts text from a PDF using pdfplumber.

The extracted text is broken into smaller chunks based on sentence count.

Each chunk is converted into a vector embedding using a sentence transformer model.

FAISS is used to store the embeddings and retrieve the most relevant chunks when a question is asked.

The retrieved chunks are sent along with the user’s query to a chosen LLM backend (Groq or local Ollama).

The model generates an answer using only the provided document context and page references.

Features

Works with any text-based PDF

Supports both online and offline language models

Prevents hallucination by forcing responses to rely on retrieved context

Includes page number citations in the output

Uses FAISS for fast similarity search

Provides real-time question answering in a command line interface

Requirements

Python 3.9 or higher

SentenceTransformers

FAISS

pdfplumber

Groq API key (if using Groq mode)

Ollama installed (if using local mode)

Setup Instructions

Install dependencies using pip install -r requirements.txt.

Place the PDF you want to query in the data folder.

Update the file path in the script if needed.

Set the backend in the script to either:

BACKEND = "groq"

BACKEND = "local"

Run the script using python main.py.

Example Questions

What is this document about?

Summarize key points from the introduction.

What methodology is described?

What treatment or conclusion does the document present?


Future Enhancements

Add support for OCR with Tesseract

Add a graphical interface using Streamlit

Add multi-document support

Add chat history and memory

Improve chunking and preprocessing

Conclusion

This project demonstrates a complete working example of a modern RAG system built from scratch. It shows how embeddings and vector search can be combined with LLMs to answer questions based on document content rather than model memory. This approach improves reliability, reduces hallucination, and enables scalable document intelligence applications.