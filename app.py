import openai
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain import OpenAI
import json
import base64


api_key = "something"

def prediction_pipeline(text):
    print("inside prediction pipeline")
    text_splitter = CharacterTextSplitter(seperator = '\n', chunk_size = 1000, chunk_overlap = 20)
    
    text_chunks = text_splitter.split_text(text)
    
    print(len(text_chunks))
    
    llm = OpenAI(openai_api_key = api_key)
    
    docs = [Document(page_content=t) for t in text_chunks]
    
    chain = load_summarize_chain(llm = llm, chain_type = 'map_reduce', verbose = True)
    
    summary = chain.run(docs)
    
    return summary