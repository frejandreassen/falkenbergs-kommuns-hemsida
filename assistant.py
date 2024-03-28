import streamlit as st
import requests
import pandas as pd
from openai import OpenAI
from scipy.spatial.distance import cosine
import ast
from streamlit_star_rating import st_star_rating

client = OpenAI(api_key=st.secrets["openai_api_key"])
headers = {"Content-Type": "application/json"}
api_url = "https://nav.utvecklingfalkenberg.se/items/falkenbergs_kommuns_hemsida"
params = {"access_token": st.secrets["directus_token"]}

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

# Define the function to update the rating


# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

def find_top_similar_texts(query_embedding, df, top_n=3, similarity_threshold=0.5):
    # Calculate similarities for each row in the dataframe
    similarities = df['embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
    # Filter results by the similarity threshold
    filtered_similarities = similarities[similarities >= similarity_threshold]
   
    if filtered_similarities.empty:
        return pd.DataFrame(), []  # Return empty DataFrame and list if no similarities meet the threshold
    # Get top N indices based on the filtered similarities
    top_idxs = filtered_similarities.nlargest(top_n).index
    print(filtered_similarities.nlargest(top_n))
    return df.iloc[top_idxs], filtered_similarities.loc[top_idxs].tolist()

# Streamlit UI
st.title("Fråga Falkenbergs kommuns webbplats")
with st.form(key='user_query_form', clear_on_submit=True):
    user_input = st.text_input("Vad letar du efter?", key="user_input")
    st.caption("Svaren genereras av en AI-bot, som kan begå misstag. Frågor och svar lagras i utvecklingssyfte. Skriv inte personuppgifter i fältet.")
    submit_button = st.form_submit_button("Sök")

if submit_button and user_input:
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
    
    Här är information som skulle kunna vara till hjälp för att hjälpa användaren:

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
    
    
    Hjälp användaren att få svar på sin fråga.
    Redovisa endast om dokumenten är relevant. 
    Om du använder dokument, hänvisa med länk till källan. 
    Svara på samma språk som användarinput.
    """
    
    st.markdown(f'#### {user_input}')
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
            st.session_state['response_completed'] = False
            if chunk.choices[0].finish_reason == "stop": 
                message_placeholder.markdown(full_response)
                st.session_state['response_completed'] = True
                break
            full_response += chunk.choices[0].delta.content
            message_placeholder.markdown(full_response + "▌")
    
    # POST request to save initial response
    data = {"prompt": user_input, "response": full_response}
    response = requests.post(api_url, json=data, headers=headers, params=params)

    
    if response.status_code == 200:
        response_data = response.json()
        st.session_state['record_id'] = response_data['data']['id']  # Save the record ID for later update
        # st.success("Tack för din feedback!")
    else:
        st.error("Något gick fel. Försök igen senare.")

if 'response_completed' in st.session_state and st.session_state['response_completed']:

    with st.form(key='user_feedback_form', clear_on_submit=True):
        stars = st_star_rating("Hur nöjd är du med svaret", maxValue=5, defaultValue=3, key="rating")
        user_feedback = st.text_area("Vad var bra/mindre bra?")
        feedback_submit_button = st.form_submit_button("Skicka")

    if feedback_submit_button and user_input:
        if 'record_id' in st.session_state and st.session_state['record_id']:
            update_data = {"user_rating": stars, "user_feedback": user_feedback}
            update_url = f"{api_url}/{st.session_state['record_id']}"
            headers = {"Content-Type": "application/json"}
            params = {"access_token": st.secrets["directus_token"]}
            print(update_url)
            
            update_response = requests.patch(update_url, json=update_data, headers=headers, params=params)

            if update_response.status_code == 200:
                st.success("Tack för din feedback!")
                st.session_state['response_completed'] = False
                st.rerun()
            else:
                st.error("Något gick fel. Tack för din feedback!")
                # st.rerun()
                # Here you might consider logging the error or notifying an administrator