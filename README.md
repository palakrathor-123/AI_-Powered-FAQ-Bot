# 🤖 AI-Powered FAQ Bot

## Project Overview
AI-Powered FAQ Bot is an intelligent chatbot that answers user questions using Semantic Search instead of simple keyword matching. The system understands the meaning of the user's query by generating sentence embeddings with Sentence Transformers and retrieves the most relevant answer from the FAQ dataset.

## Features
- Semantic Search using Sentence Transformers
- FAQ Question Answering
- Cosine Similarity Matching
- FAISS-based Fast Similarity Search
- Confidence Score for Answers
- Fallback Response for Unmatched Queries
- User-Friendly Streamlit Interface
- Optional Sentiment Analysis
- Optional Translation Support
- Optional Conversation Summarization

## Technologies Used
- Python
- Streamlit
- Pandas
- Sentence Transformers
- Hugging Face Transformers
- FAISS
- Scikit-learn
- NumPy

## Workflow
1. Load the FAQ dataset.
2. Preprocess the questions.
3. Generate embeddings for all FAQ questions.
4. Store embeddings using FAISS.
5. Accept the user's question.
6. Generate embedding for the query.
7. Compare with stored FAQ embeddings using cosine similarity.
8. Return the most relevant answer with a confidence score.
9. Display a fallback response if no relevant answer is found.

## Dataset
The project uses a CSV-based FAQ dataset containing:
- Question
- Answer
- Category

## Output
The chatbot accepts user questions through a Streamlit interface and returns the most relevant FAQ answer along with its confidence score. If no suitable answer is found, it displays a fallback message.

## Future Improvements
- Voice Input
- Speech Output
- Database Integration
- Multi-language Support
- User Authentication
- Dynamic FAQ Management

## Author
Palak Rathor