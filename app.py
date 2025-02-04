import streamlit as st
from preprocess import process_documents
from qa_chain import initialize_qa_chain
import os


def main():
    st.title("📄 Intelligent Document QA System")

    # Initialize session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    # Sidebar for document upload
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=["pdf", "txt", "docx", "pptx"],
            accept_multiple_files=True
        )

    if uploaded_files:
        with st.spinner("Processing documents..."):
            texts = process_documents(uploaded_files)
            st.session_state.qa_chain = initialize_qa_chain(texts)
    else:
        st.warning("Please upload documents first")

    question = st.chat_input("Enter your question:", key="question_input")

    if question and st.session_state.qa_chain:
        with st.spinner("Analyzing documents..."):
            result = st.session_state.qa_chain.invoke({"input": question})
            # st.subheader("Source Documents")
            # for i, doc in enumerate(result["source_documents"], 1):
            #     st.markdown(
            #         f"**Source {i}** (Page {doc.metadata.get('page', 'N/A')})")
            #     st.caption(doc.page_content[:500] + "...")
            st.markdown(result['answer'])
    elif question:
        st.warning("Please process documents first")


if __name__ == "__main__":
    main()
