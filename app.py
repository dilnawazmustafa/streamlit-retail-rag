
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI

st.title("Retail Sales RAG Chatbot")

uploaded_file = st.file_uploader("Upload your retail sales CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Here's a preview of your data:")
    st.dataframe(df.head())

    question = st.text_input("Ask a question about your data:")

    def retrieve_rows(question, df, n=10):
        terms = question.lower().split()
        mask = df.apply(lambda row: any(term in str(row).lower() for term in terms), axis=1)
        matches = df[mask]
        return matches.head(n)

    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if openai_api_key:
        llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

        def ask_csv(question, df):
            context_df = retrieve_rows(question, df)
            context = context_df.to_csv(index=False)
            prompt = (
                f"Given this retail sales data:\n{context}\n\n"
                f"Answer this question as a helpful analyst: {question}"
            )
            result = llm.invoke(prompt)
            return result.content

        if question:
            with st.spinner("Thinking..."):
                answer = ask_csv(question, df)
                st.markdown(f"**Answer:** {answer}")
