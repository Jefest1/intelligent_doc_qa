from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
import jinja2
import os

# For Claude, AWS Titan Embeddings work well, or Cohere embeddings
# Here's implementation with Cohere (requires COHERE_API_KEY in .env)
from langchain_cohere import CohereEmbeddings

def load_prompt_template():
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("prompts/"))
    template = env.get_template("qa_prompt.jinja2")
    return PromptTemplate(
        template=template.render(),
        input_variables=["documents", "question"],
        template_format="jinja2"
    )

def initialize_qa_chain(texts):
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    
    vector_store = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory="./chroma_db"
    )
    vector_store.persist()
    
    llm = ChatAnthropic(
        model_name="claude-3-sonnet-20240229",
        temperature=0,
        max_tokens=1000
    )
    
    qa_prompt = load_prompt_template()
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 3}
        ),
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )