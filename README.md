# 🎬 Netflix Movie Recommendation System

A content-based movie recommendation web application built using Python, Machine Learning, Streamlit, and the OMDb API.

## 🚀 Features

- 🔍 Search for movies
- 🎯 Content-based movie recommendations
- 🤖 TF-IDF vectorization
- 📊 Cosine similarity
- 🖼️ Movie posters
- ⭐ IMDb ratings
- 🎭 Genre information
- 🎬 Director and cast information
- 📝 Movie plot
- 🔥 Popular movies
- 📈 Trending movies
- ⚡ Cached recommendations and API responses
- 🎨 Netflix-inspired Streamlit interface

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- REST API
- OMDb API
- TF-IDF
- Cosine Similarity

## 🧠 Machine Learning Approach

This project uses a content-based recommendation approach.

Movie information such as genres, overview, cast, director, and other textual features are combined into movie tags.

TF-IDF converts these text features into numerical vectors.

Cosine similarity is then used to measure the similarity between movies.

When a user searches for or selects a movie, the system finds movies with similar content and recommends them.

## Project Structure

```text
Movie Recommendation System/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── movie_list.pkl
│   └── similarity.pkl
│
├── assets/
│   └── poster_placeholder.png
│
├── .streamlit/
│   └── secrets.toml
│
└── .gitignore