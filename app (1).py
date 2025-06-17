
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI

st.title("Retail Sales RAG Chatbot")

# Upload CSV section
uploaded_file = st.file_uploader("Upload your retail sales CSV", type="csv")

# Show the dataframe preview if a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Here's a preview of your data:")
    st.dataframe(df.head())

    # Input OpenAI key and question
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    question = st.text_input("Ask a question about your data:")
    submit = st.button("Submit")

    def retrieve_rows(question, df, n=10):
        terms = question.lower().split()
        mask = df.apply(lambda row: any(term in str(row).lower() for term in terms), axis=1)
        matches = df[mask]
        return matches.head(n)

    def ask_csv(question, df, openai_api_key):
        llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
        context_df = retrieve_rows(question, df)
        context = context_df.to_csv(index=False)
        prompt = (
            f"Given this retail sales data:\n{context}\n\n"
            f"Answer this question as a helpful analyst: {question}"
        )
        result = llm.invoke(prompt)
        return result.content

    # Only run the query when user clicks Submit and both key/question are provided
    if openai_api_key and question and submit:
        with st.spinner("Thinking..."):
            try:
                answer = ask_csv(question, df, openai_api_key)
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"Error: {e}")
