from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from constants import *

# the prompt: we will be changing this soon
prompt = "hello world!"

# Note: we must use the same embedding model that we used when uploading the docs
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Querying the vector database for "relevant" docs then create a retriever
document_vectorstore = PineconeVectorStore(index_name="pineidx", embedding=embeddings)
retriever = document_vectorstore.as_retriever()

# create a context by using the retriever and getting the relevant docs based on the prompt
context = retriever.get_relevant_documents(prompt)
# show the thought process by looping over all relevant docs, showing the source and the content
for doc in context:
    print(f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n\n")
    print("__________________________")

# build a prompt template using the query and the context and build the prompt with context


# Asking the LLM for a response from our prompt with the provided context using CatOpenAI and invoking it
# Then print the results content
