from langchain_groq.chat_models import ChatGroq  
import streamlit as st 
import pandas as pd 
from pandasai import SmartDataframe
import os

# Configure the OpenAI model with your API key
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-70b-versatile")

st.title("Data analysis with PandasAI")

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3))

    # Set up the SmartDataframe with OpenAI as the LLM
    df = SmartDataframe(data, config={"llm": model})
    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                try:
                    # Using OpenAI to generate the response based on your prompt
                    response = df.chat(prompt)
                    
                    # Check if the response is a DataFrame and display it as a table
                    if isinstance(response, pd.DataFrame):
                        st.write("Generated DataFrame:")
                        st.dataframe(response)
                    else:
                        st.write(response)
                except Exception as e:
 
                   st.error(f"An error occurred: {e}")

