import streamlit as st
import requests
from openai import OpenAI
from streamlit_star_rating import st_star_rating
from qdrant_client import QdrantClient

# collection_name="FalkenbergsKommunsHemsida"
collection_name="FalkenbergsKommunsHemsida_1000char_chunks"
headers = {"Content-Type": "application/json"}
api_url = "https://nav.utvecklingfalkenberg.se/items/falkenbergs_kommuns_hemsida"
params = {"access_token": st.secrets["directus_token"]}
qdrant_api_key = st.secrets['qdrant_api_key']
qdrant_url = 'https://qdrant.utvecklingfalkenberg.se'
qdrant_client = QdrantClient(url=qdrant_url, port=443, https=True, api_key=qdrant_api_key)
openai_client = OpenAI(api_key=st.secrets["openai_api_key"])
GPT_MODEL = "gpt-4-turbo"

def generate_embeddings(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    # Accessing the embedding properly based on OpenAI documentation
    return response.data[0].embedding

def search_collection(qdrant_client, collection_name, user_query_embedding):
    response = qdrant_client.search(
        collection_name=collection_name,
        query_vector=user_query_embedding,
        limit=5,
        with_payload=True
    )
    return response  # Adjust this line if the structure is different

# Streamlit UI
st.title("Fråga Falkenbergs kommuns webbplats")
with st.form(key='user_query_form', clear_on_submit=True):
    user_input = st.text_input("Vad letar du efter?", key="user_input", placeholder="När börjar sommarlovet för Tullbroskolan?")
    st.caption("Svaren genereras av en AI-bot, som kan begå misstag. Svaren baseras på tillgänglig information på kommunens hemsida. Frågor och svar lagras i utvecklingssyfte. Skriv inte personuppgifter i fältet.")
    submit_button = st.form_submit_button("Sök")

if submit_button and user_input:
    # Generate embedding for user input
    user_embedding = generate_embeddings(user_input)

    # Find the top two most similar text chunks
    search_results = search_collection(qdrant_client, collection_name, user_embedding)

    similar_texts = [
        {"chunk": result.payload['chunk'], "title": result.payload['title'], "url": result.payload['url'], "score": result.score}
        for result in search_results
    ]
    print(similar_texts)
    # Prepare the prompt for GPT-4 in Swedish
    instructions_prompt = f"""
    Användarinput: {user_input}
    Du är en hjälpsam assistent som hjälper användaren att hitta information om Falkenbergs kommun. 
    
    Här är information som skulle kunna vara till hjälp för att hjälpa användaren:

    Dokument:
    {similar_texts[0]['chunk']}
    URL: {similar_texts[0]['url']}
    Likhetsscore: {similar_texts[0]['score']}
    
    Dokument:
    {similar_texts[1]['chunk']}
    URL: {similar_texts[1]['url']}
    Likhetsscore: {similar_texts[1]['score']}

    Dokument:
    {similar_texts[2]['chunk']}
    URL: {similar_texts[2]['url']}
    Likhetsscore: {similar_texts[2]['score']}

        Dokument:
    {similar_texts[3]['chunk']}
    URL: {similar_texts[3]['url']}
    Likhetsscore: {similar_texts[3]['score']}

        Dokument:
    {similar_texts[4]['chunk']}
    URL: {similar_texts[4]['url']}
    Likhetsscore: {similar_texts[4]['score']}
    
    
    Hjälp användaren att få svar på sin fråga.
    Redovisa endast om dokumenten är relevant. 
    Om du använder dokument, hänvisa med länk till källan. 
    Reply in the same language as: {user_input}.
    """
    
    st.markdown(f'#### {user_input}')
    # Stream the GPT-4 reply
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        completion = openai_client.chat.completions.create(
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