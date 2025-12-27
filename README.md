🎬 Movie Recommendation Engine

A Python-based movie recommendation engine that suggests movies based on user selections and movie features such as genres, release year, and title similarity. The project demonstrates data preprocessing, feature engineering, and KNN-based recommendations.

📌 Project Overview

This project allows users to:

Get random movie recommendations

Search movies by title

Select movies and receive personalized recommendations

Display movies with basic info, IMDB links, and overview snippets

The recommendation engine combines genre, year, and title features using TF-IDF and MultiLabelBinarizer and leverages K-Nearest Neighbors (KNN) to suggest similar movies.

## Demo Video 📺

Watch the demo of the project here:  
https://youtu.be/f3U--m8XQSA?si=-tmPe7QMJaxH_ht0

🛠️ Technologies Used

Python 3

Pandas & NumPy

Scikit-learn (NearestNeighbors, MultiLabelBinarizer, TfidfVectorizer)

Random sampling for recommendations

📂 Project Structure
├── movies.csv               # Movie dataset (ID, title, genres)
├── movies_description.csv   # Movie descriptions (title, overview)
├── movie_recommender.py     # Main recommendation engine
├── README.md                # Documentation

🔍 Key Features
1️⃣ Data Loading & Preprocessing

Extract movie year from title

Clean titles and split genres into lists

2️⃣ Feature Engineering

Convert genres to binary features

Normalize year feature

Extract TF-IDF vectors from titles

3️⃣ Recommendations

Random recommendations if no movies are selected

KNN-based personalized recommendations based on selected movies

Ensures already selected movies are excluded from recommendations

4️⃣ Search & User Interaction

Search movies by title

Display up to 12 search results

Show user-selected movies with basic info and IMDB poster URLs

5️⃣ User-Friendly CLI

Select movies using movie IDs

Search movies by title

Clear search results and return to recommendations

▶️ How to Run

Clone the repository:

git clone https://github.com/your-username/movie-recommendation-engine.git


Install dependencies:

pip install pandas numpy scikit-learn


Run the recommendation engine:

python movie_recommender.py

🎯 Sample Usage
🎲 RANDOM RECOMMENDATIONS
🎬 ID: 1 - Toy Story
   📅 Year: 1995
   🎭 Genres: Animation|Adventure|Comedy
   ⭐ IMDB Rating: 7.5/10
   🖼️ Poster: https://www.imdb.com/title/tt0000001/mediaviewer/
   📖 Overview: A story about...


Select a movie: Enter the movie ID

Search by title: Enter a search term

Get personalized recommendations: Engine will update based on selections

🚀 Future Improvements

Integrate actual IMDB ratings

Build GUI/Web interface

Add user preference learning for better recommendations

Include more advanced content-based and collaborative filtering methods

👩‍💻 Author

Jayanti Acharya
Computer Science Student
