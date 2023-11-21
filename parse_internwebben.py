import pandas as pd
from bs4 import BeautifulSoup

# Function to split text into chunks with an overlap
def split_text(text, chunk_size=200, overlap=25):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks


# Load the CSV file
df = pd.read_csv("internwebben.csv")

# Initialize a list to store the results
results = []

# Iterate through each row in the dataframe
for index, row in df.iterrows():
    # Extracting necessary fields
    title = row['Title']
    url = "https://intranet.falkenberg.se" + row['URL']
    content_html = row['Content']

    # Check if content_html is NaN (missing value)
    if pd.isna(content_html):
        # Handle the missing value (you can choose to skip or use a placeholder)
        continue  # This will skip the current iteration and move to the next row

    # Extracting text from HTML content
    soup = BeautifulSoup(content_html, 'html.parser')
    content_text = soup.get_text(separator='\n')

    # Split the content text into chunks
    chunks = split_text(content_text)
    total_chunks = len(chunks)  # Total number of chunks for this document

    for chunk_number, chunk in enumerate(chunks, start=1):
        chunk_title = f"{title} - del {chunk_number}/{total_chunks}"
        results.append({
            'title': title,
            'url': url,
            'document': chunk_title + "\n" + chunk
        })
# Create a DataFrame to store the results
output_df = pd.DataFrame(results)

# Save the results to a new CSV file
output_df.to_csv("output_processed_internwebben.csv", index=False)
