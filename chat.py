import streamlit as st
import pandas as pd
from openai import OpenAI
from scipy.spatial.distance import cosine
import ast

client = OpenAI(api_key=st.secrets["openai_api_key"])


# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4-1106-preview"

# Load the CSV file with embeddings
@st.cache_data
def load_data():
    df1 = pd.read_csv('output_with_embeddings_part1.csv')
    df2 = pd.read_csv('output_with_embeddings_part2.csv')
    df1['embedding'] = df1['embedding'].apply(ast.literal_eval)
    df2['embedding'] = df2['embedding'].apply(ast.literal_eval)
    df = pd.concat([df1, df2], ignore_index=True)
    return df

df = load_data()

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Function to find the top two most similar text chunks
def find_top_similar_texts(query_embedding, df, top_n=3):
    similarities = df['embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
    top_idxs = similarities.nlargest(top_n).index
    return df.iloc[top_idxs], similarities[top_idxs].tolist()

# Streamlit UI
st.title("Fråga Falkenbergs Kommuns Hemsida")

user_input = st.text_input("Vad letar du efter?")

if user_input:
    # Generate embedding for user input
    response = client.embeddings.create(
        model= EMBEDDING_MODEL, 
        input= f'{user_input}'
        )

    user_embedding = response.data[0].embedding

    # Find the top two most similar text chunks
    similar_texts, similarities = find_top_similar_texts(user_embedding, df)
    
    # Prepare the prompt for GPT-4 in Swedish
    instructions_prompt = f"""
    Användarinput: {user_input}
    Du är en hjälpsam assistent som hjälper användaren att hitta information om Falkenbergs kommun. 
    Baserat på dokumenten som hittades, här är informationen för att hjälpa användaren:

    Dokument:
    {similar_texts.iloc[0]['document']}
    URL: {similar_texts.iloc[0]['source']}
    Likhetsscore: {similarities[0]}
    
    Dokument:
    {similar_texts.iloc[1]['document']}
    URL: {similar_texts.iloc[1]['source']}
    Likhetsscore: {similarities[1]}

    Dokument:
    {similar_texts.iloc[2]['document']}
    URL: {similar_texts.iloc[2]['source']}
    Likhetsscore: {similarities[2]}
    
    
    Hjälp användaren att få svar på sin fråga. Redovisa var du har fått informationen som du baserar ditt svar på, och hänvisa med länk till källan. Redovisa endast om dokumenten är relevant. Svara på samma språk som användarinput.
    """
    
    # Stream the GPT-4 reply
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        completion = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "system", "content": instructions_prompt}],
            stream=True
        )
        for chunk in completion:
            if chunk.choices[0].finish_reason == "stop": 
                message_placeholder.markdown(full_response)
                break
            full_response += chunk.choices[0].delta.content
            message_placeholder.markdown(full_response + "▌")