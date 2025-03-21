import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from functools import partial
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
import gradio as gr

# number of movies to be retrieved
total_movies = 50

# function used to obtain preference information from the user
def load_user_preferences(user_id, filename):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
            for user in data:
                if user["user_id"] == user_id:
                    return user["preferences"]
            return None
    except FileNotFoundError:
        return None

# function used to format user preferences in order to use them in the prompt
def format_user_preferences(preferences):
    if not preferences:
        return "No preferences available."

    formatted_preferences = []
    if "favorite_genres" in preferences:
        formatted_preferences.append(f"Favorite Genres: {', '.join(preferences['favorite_genres'])}")
    if "disliked_genres" in preferences:
        formatted_preferences.append(f"Disliked Genres: {', '.join(preferences['disliked_genres'])}")
    return "\n".join(formatted_preferences)

# function used to do a genre mapping, by relating the id to the gender name
def fetch_genre_mapping(tmdb_token):
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={tmdb_token}&language=en-US"
    response = requests.get(url).json()
    return {genre['id']: genre['name'] for genre in response['genres']}

# function used to fetch movie data
def fetch_movie_data(tmdb_token, num_movies):
    url = f"https://api.themoviedb.org/3/movie/popular?api_key={tmdb_token}&language=en-US&page=1"
    response = requests.get(url).json()
    return response['results'][:num_movies]

# Function used to perform data preprocessing, aggregating the information of movies and their genre
def preprocess_movie_data(movies, genre_mapping):
    processed_movies = []
    for movie in movies:
        # if genre_ids is not empty
        if movie['genre_ids']:
            genres = [genre_mapping[genre_id] for genre_id in movie['genre_ids']]
        # if genre_ids is empty, use a default value
        else: 
            genres = ["Unknown"]
        processed_movies.append({
            "id": movie['id'],
            "title": movie['title'],
            "description": movie['overview'],
            "genres": ", ".join(genres)
        })
    return processed_movies

# function used to interface with llm to get advice from movies
def get_llm_response(hugging_token, tmdb_token, user_input, user_id):
    # login to huggingface
    login(token = hugging_token)
   
    # get the user's preferences and format them
    preferences = load_user_preferences(user_id, "preferences.json")
    formatted_preferences = format_user_preferences(preferences)

    # get the movies information from TMDB APIs and format it
    genre_mapping = fetch_genre_mapping(tmdb_token)
    movies = fetch_movie_data(tmdb_token, total_movies)
    processed_movies = preprocess_movie_data(movies, genre_mapping)
    
    # init embedding model
    transformer_path = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(transformer_path)
    Settings.embed_model = HuggingFaceEmbeddings(model_name = transformer_path)
    
    # Generate embeddings for movie data
    movie_embeddings = []
    for movie in processed_movies:
        text = f"{movie['title']} {movie['description']} {movie['genres']}"
        embedding = embedding_model.encode(text)
        movie_embeddings.append((movie['id'], embedding))

    # convert movie data to LlamaIndex TextNode objects
    nodes = []
    for movie, (movie_id, embedding) in zip(processed_movies, movie_embeddings):
        node = TextNode(
            text = f"{movie['title']} {movie['description']} {movie['genres']}",
            embedding = embedding.tolist(),
            metadata={
                "id": movie['id'],
                "title": movie['title'],
                "description": movie['description'],
                "genres": movie['genres'],
            }
        )
        nodes.append(node)

    # init vector db
    vector_store = SimpleVectorStore()

    # Create a storage context
    storage_context = StorageContext.from_defaults(vector_store = vector_store)

    # Create an index
    index = VectorStoreIndex(nodes = nodes, storage_context=storage_context)

    # uses llama index to get the five similar movies with respect to the input
    retriever = index.as_retriever(similarity_top_k = 5)
    retrieved_movies = retriever.retrieve(user_input)
    retrieved_text = "\n".join([node.text for node in retrieved_movies])

    # define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["user_input", "user_preferences", "retrieved_text"],
        template= "Based on the following user preferences: {user_preferences} And the following user input: {user_input} Recommend a movie and explain why from these options: {retrieved_text}"
    )

    # format the prompt
    prompt = prompt_template.format(user_input = user_input, user_preferences = formatted_preferences, retrieved_text = retrieved_text)

    # init client for making requests to HuggingFace Inference API
    client = InferenceClient(token=hugging_token)
    
    # query the model to get movie suggestions
    response = client.chat_completion(
        model = "mistralai/Mistral-7B-Instruct-v0.2",
        messages = [{"role": "user", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"]
    
if __name__ == "__main__":
    # loads environment variables from the “env” file
    load_dotenv(find_dotenv())
    
    # checks that all tokens are available
    token_list = ["HUGGINGFACEHUB_API_TOKEN", "TMDB_API_KEY", "OPENAI_API_KEY"]
    for token in token_list:
        if not os.getenv(token):
            print(f"ERROR: {token} not found\n")
            sys.exit()
    
    # statically add tokens arguments to query with the llm
    llm_function = partial(get_llm_response, os.getenv("HUGGINGFACEHUB_API_TOKEN"), os.getenv("TMDB_API_KEY"))
   
    # use gradio for creating a simple graphical interface with which to interact with the llm
    iface = gr.Interface(
        fn = llm_function,
        inputs = [
            gr.Textbox(label="Enter your prompt"),
            gr.Textbox(label="Enter your user ID"),
        ],
        outputs = gr.Textbox(label = "LLM Response"),
        flagging_mode = "never",
        title = "Movie Recommendation System",
        description = "Enter a prompt and your user ID to get personalized movie recommendations."
    )
    iface.launch()