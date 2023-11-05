import asyncio
import json
import torch
import logging
import aiosqlite
from llama_cpp import Llama
import weaviate
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the LLaMA model
llm = Llama(
    model_path="llama-2-7b-chat.ggmlv3.q8_0.bin",
    n_gpu_layers=-1,
    n_ctx=3900,
)

# Initialize the ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=3)

# Initialize the Weaviate client
client = weaviate.Client(
    url="https://tacoai-9pf21pxx.weaviate.network",
)

# Database initialization function
async def initialize_db():
    async with aiosqlite.connect("movie_frames.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS POOLDATA (
                frame_num INTEGER PRIMARY KEY,
                frame_text TEXT,
                summary TEXT
            );
        """)
        await db.commit()

# Function to insert data into the SQLite database
async def insert_into_db(frame_num, frame_text, summary):
    async with aiosqlite.connect("movie_frames.db") as db:
        try:
            await db.execute("INSERT INTO POOLDATA (frame_num, frame_text, summary) VALUES (?, ?, ?)",
                             (frame_num, frame_text, summary))
            await db.commit()
        except aiosqlite.IntegrityError as e:
            if 'UNIQUE constraint failed' in str(e):
                logger.warning(f"Frame number {frame_num} already exists. Skipping insert.")
            else:
                raise


# Function to generate and summarize a movie frame
async def generate_and_summarize_frame(last_frame, frame_num, frames):
    """Generates a new movie frame based on the last frame and summarizes it using LLaMA."""
    # Generate a prompt for the LLaMA model to continue the storyline.
    continuation_prompt = f"Continue the storyline based on the last scene: '{last_frame}'."

    # Generate a new movie frame.
    try:
        # Call the Llama object directly with the prompt and parameters.
        continuation_response = llm(continuation_prompt, max_tokens=200)
        new_frame = continuation_response['choices'][0]['text'] if continuation_response['choices'] else None
        if new_frame:
            # Now, generate a summary for the new frame.
            summary_prompt = f"Summarize the following storyline: '{new_frame}'."
            # Call the Llama object directly with the prompt and parameters.
            summary_response = llm(summary_prompt, max_tokens=50)
            summary = summary_response['choices'][0]['text'] if summary_response['choices'] else "Summary not generated."

            frames[f"{frame_num}"] = new_frame
            # Insert the new frame and its summary into the SQLite database.
            await insert_into_db(frame_num, new_frame, summary)
            # Prepare the movie frame data for Weaviate insertion.
            movie_frame = {"frame_text": new_frame, "summary": summary}
            # Insert the new frame and its summary into Weaviate.
            await insert_into_weaviate(movie_frame)
            return new_frame
        else:
            logger.error(f"No valid frame generated for frame number {frame_num}.")
            return None
    except Exception as e:
        logger.error(f"Failed to generate frame {frame_num}: {e}")
        return None
    
# Assuming that the Weaviate client has been initialized and the schema has been set up as shown previously.

async def insert_into_weaviate(movie_frame):
    try:
        # Construct the data object for Weaviate according to the defined schema.
        weaviate_object = {
            "frame_text": movie_frame["frame_text"],
            "summary": movie_frame["summary"]
        }
        
        # Insert the object into Weaviate.
        # The vectorization happens automatically based on the class schema.
        # The 'class_name' parameter is required for the 'create' method.
        # Run the synchronous create method in the executor.
        await loop.run_in_executor(
            executor, 
            client.data_object.create,
            weaviate_object, 
            "MovieFrame",
            None  # UUID can be None to let Weaviate generate it, or you can provide one.
        )
        
        logger.info(f"Successfully inserted movie frame into Weaviate: {movie_frame}")
    except Exception as e:
        logger.error(f"Failed to insert data into Weaviate: {e}")



# Main function to start generating the movie
async def start_movie(topic):
    # Initialize the SQLite database.
    await initialize_db()

    # Check if the Weaviate class for the movie frames already exists.
    try:
        # Assuming client.schema.get() is synchronous based on the context.
        classes = client.schema.get()
        class_names = [cls['class'] for cls in classes['classes']] if 'classes' in classes else []
        if "MovieFrame" not in class_names:
            # Create a Weaviate class for the movie frames with vector indexing.
            # Assuming client.schema.create_class() is synchronous based on the context.
            client.schema.create_class(
                {
                    "class": "MovieFrame",
                    "properties": [
                        {
                            "name": "frame_text",
                            "dataType": ["text"],
                            "vectorizer": "text2vec-transformers"
                        },
                        {
                            "name": "summary",
                            "dataType": ["text"],
                            "vectorizer": "text2vec-transformers"
                        }
                    ],
                    "vectorIndexType": "hnsw",
                    "vectorizer": "text2vec-transformers"
                }
            )
    except weaviate.exceptions.SchemaValidationException as e:
        # Handle the specific case where the class already exists
        if 'class already exists' in str(e):
            logger.info("Weaviate class 'MovieFrame' already exists.")
        else:
            logger.error(f"Failed to create Weaviate class: {e}")
            return
    except Exception as e:
        logger.error(f"An error occurred while checking or creating the Weaviate class: {e}")
        return

    # Generate the opening scene for the movie.
    try:
        # Assuming llm.generate() is synchronous based on the context.
        initial_prompt_response = llm(
            f"As an AI specialized in creating scripts, generate the opening scene for a movie about {topic}.",
            max_tokens=700
        )
        if initial_prompt_response is None or 'choices' not in initial_prompt_response:
            logger.error("LLM did not return a valid response.")
            return
        initial_prompt = initial_prompt_response['choices'][0]['text']
    except Exception as e:
        logger.error(f"Failed to generate initial prompt: {e}")
        return

    # Start generating the movie script.
    frames = {"0": initial_prompt}
    last_frame = initial_prompt

    for i in range(1, 100):
        frame_num = i * 10
        last_frame = await generate_and_summarize_frame(last_frame, frame_num, frames)

    # Save the generated movie frames to a JSON file.
    sanitized_topic = ''.join(e for e in topic if e.isalnum())[:50]
    with open(f"{sanitized_topic}_movie_frames.json", "w") as f:
        json.dump(frames, f, indent=4)

    return f"Movie script about {topic} started and 10 frames generated. Saved to {sanitized_topic}_movie_frames.json"

# Run the script
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_movie("A movie about three hackers who build a Quantum Language Model"))

    del llm
    torch.cuda.empty_cache()
