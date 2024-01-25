import pandas as pd
import requests
import os
from PyPDF2 import PdfReader
import math

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

# Function to download a PDF and return its text content
def download_and_extract_pdf_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        reader = PdfReader("temp.pdf")
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        os.remove("temp.pdf")
        return text
    else:
        print(f"Failed to download PDF from {url}")
        return ""

# Load the CSV file
df = pd.read_csv("Falkenbergs_kommun_202401.csv")

# Initialize a list to store the results
results = []

# Iterate through each row in the dataframe
for index, row in df.iterrows():
    category = row['område']
    title = row['under_under_område']
    url = row['under_under_område-href']

    # Process text content
    content = row['content']
    under_content = row['under_content'] if not pd.isna(row['under_content']) else ""
    full_text = f"{content}\n{under_content}"
    chunks = split_text(full_text)
    for chunk in chunks:
        results.append({
            'category': category,
            'title': title,
            'source': url,
            'document': chunk
        })

    # Process PDFs
    if not pd.isna(row['pdfs-href']):
        pdf_url = row['pdfs-href']
        pdf_text = download_and_extract_pdf_text(pdf_url)
        pdf_chunks = split_text(pdf_text)
        for chunk in pdf_chunks:
            results.append({
                'category': category,
                'title': title,
                'source': pdf_url,
                'document': chunk
            })

# Create a DataFrame to store the results
output_df = pd.DataFrame(results)

# Save the results to a new CSV file
output_df.to_csv("output_file.csv", index=False)
