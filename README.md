# movie-director-qml

The provided Python script is an asynchronous program designed to generate and store a sequence of creative text outputs, referred to as "movie frames," using a LLaMA model. It interacts with a SQLite database and a Weaviate instance for data storage. Here's a summary with markdown formatting for clarity:

# Demo : ![image](https://github.com/graylan0/movie-director-qml/assets/34530588/0ca86730-f473-4234-a807-4f7baa03f230)

![image](https://github.com/graylan0/movie-director-qml/assets/34530588/db925f13-1e8e-448f-bf86-538973b605d8)


### Script Overview

**Purpose**: To create a movie script based on a given topic by generating and summarizing text frames.

**Components**:
- **LLaMA Model**: Used for generating creative text and summaries.
- **SQLite Database**: Stores the generated frames and summaries.
- **Weaviate**: A vector database for storing and indexing text data.

### Process Flow

1. **Initialization**:
   - Set up logging.
   - Initialize the LLaMA model, ThreadPoolExecutor, and Weaviate client.

2. **Database Setup**:
   - Create a SQLite table if it doesn't exist.

3. **Data Generation and Storage**:
   - Generate new frames using the LLaMA model.
   - Summarize the generated frames.
   - Insert data into SQLite and Weaviate.

4. **Error Handling**:
   - Log warnings for duplicate entries in SQLite.
   - Log errors for invalid frames or Weaviate insertion failures.

### Main Functionality

- **Start Movie**: Generates an opening scene and subsequent frames for a movie script about a specified topic.
- **Loop**: Iterates to create multiple frames, each time generating and summarizing a new frame based on the last.

### Execution

- Uses `nest_asyncio` to allow the asynchronous loop to run in a Jupyter notebook or similar environments.
- Cleans up by deleting the LLaMA model instance and clearing the GPU cache.

### Error Messages

- Handles specific error cases, such as "No valid frame generated" and Weaviate insertion failures, with appropriate logging.


