from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_anthropic import ChatAnthropic
import jinja2
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings
from config import settings


def load_prompt_template():
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("prompts/"))
    template = env.get_template("qa_prompt.jinja2")
    return PromptTemplate(
        template=template.render(),
        input_variables=["input", "context"],
        template_format="jinja2"
    )


def initialize_qa_chain(texts):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)

    vector_store = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory='embeddings_db'
    )

    llm = ChatAnthropic(
        model_name="claude-3-sonnet-20240229",
        max_tokens=10000
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    qa_prompt = load_prompt_template()

    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create the retrieval chain with the correct input key
    chain = create_retrieval_chain(
        retriever,
        document_chain,
    )

    return chain
