import openai
import pandas as pd
import time

openai.api_key = "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"  # Replace with your actual OpenAI API key
EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE = 1000  # Adjust as needed; note the OpenAI API has a limit on the number of tokens per request
SLEEP_TIME = 1  # in seconds

# Read the CSV file
input_file = 'output_file.csv'  # Replace with the path to your CSV file
df = pd.read_csv(input_file)

# Extract text chunks from the CSV data
text_chunks = df['document'].tolist()

# Initialize list to store embeddings
embeddings = []

# Make API requests in batches
for batch_start in range(0, len(text_chunks), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = text_chunks[batch_start:batch_end]
    print(f"Processing Batch {batch_start} to {batch_end-1}")
    
    try:
        # Make API request
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
        
        # Double-check embeddings are in the same order as input
        for i, be in enumerate(response['data']):
            assert i == be['index']
        
        # Extract and store embeddings
        batch_embeddings = [e['embedding'] for e in response['data']]
        embeddings.extend(batch_embeddings)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    # Sleep for a short duration to throttle requests
    time.sleep(SLEEP_TIME)

# Add embeddings to the original data
df['embedding'] = embeddings

# Save the data with embeddings to a new CSV file
output_file = 'output_with_embeddings.csv'
df.to_csv(output_file, index=False)

print(f"Data with embeddings has been saved to {output_file}")
