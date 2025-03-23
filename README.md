# Movie recommendation system with LLM
Simple movie recommendation system based on **Mistral** LLM with data management by **llama-index** of information from **tmdb** movie database

# Dependencies
[Python](https://www.python.org/)

[Hugging Face](https://huggingface.co/)

[Faiss](https://www.gradio.app/)

[LLama-Index](https://faiss.ai/index.html)

[Gradio](https://www.gradio.app/)

# General functioning
The information of the most popular movies are initially extracted with tmdb APIs and pre-processed to aggregate the genre information of the individual film with its title and description.
These data become the starting point for the generation of embeddings by the **all-MiniLM-L6-v2** transformer, which creates for each movie a document that will be then converted into embeddings for **Faiss**.
The similarity search exploits the faiss vector database included as backend in llama-index with **FaissVectorStore**.
After extracting information based on user input by similarity, the user input prompt is formatted with the obtained information to enrich the context and to make the LLM search more precise.
The recommendation system is enhanced by a simple preference system saved in a "**preferences.json**" file.
The file is structured in such a way that for each user, each identified by a unique identifier, likes and dislikes genres are associated, automatically including this information in the input prompt.
It is possible to interact with the LLM via a web interface defined by the gradio library.

# Token Setup
To take advantage of the application, it is necessary to set up in the project folder a **.env** file.
This file will contain the environment variables, one for each service access token, which will be automatically loaded via the **dotenv** library.

The keys required for proper operation of the application are as follows:

- HUGGINGFACEHUB_API_TOKEN
- TMDB_API_KEY
- OPENAI_API_KEY

These are responsible for accessing the models provided by HuggingFace, for accessing the tmdb movies database, and for accessing the openAI API for inference, respectively.

# Starting the application
The application is executable via the command:
<p align="center">
  python llm.py
</p>

Once this is done, to interact with the model, it is necessary to connect with a browser at the following address:
<p align="center">
  http://127.0.0.1:7860
</p>

# Result
![](https://github.com/AndreaFilippini/llm-movie-recommendation/blob/main/src/images/movies_result.png)
