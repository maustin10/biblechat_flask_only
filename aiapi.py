# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import config

# These imports are so that we can use the embeddings
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
from urllib.parse import unquote

from sentence_transformers import SentenceTransformer

import pandas as pd
import numpy as np
import textwrap
from langchain.embeddings import HuggingFaceEmbeddings


from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

api_key=config.DevelopmentConfig.OPENAI_KEY
openai.api_key = api_key 

## Define How We Will Do Embeddings
#Building embeddings using LangChain's OpenAI embedding support is fairly straightforward. We could use OpenAI with  [OpenAI api key]() or Hugging Face Sentence Transformers:

model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(model_name=model_name)


##################################################################
# Connect to the Milvus Vector Database (which is already running)
##################################################################
HOST = '143.198.12.142'
PORT = 19530
COLLECTION_NAME = 'borg_data'
DIMENSION = 768
#DIMENSION = 1536
#OPENAI_ENGINE = 'text-embedding-ada-002'

openai.api_key = "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"

INDEX_PARAM = {
    'metric_type':'L2',
    'index_type':"HNSW",
    'params':{'M': 8, 'efConstruction': 64}
}

QUERY_PARAM = {
    "metric_type": "L2",
    "params": {"ef": 64},
}

BATCH_SIZE = 1000

from pymilvus import connections, utility, FieldSchema, Collection, CollectionSchema, DataType

# Connect to Milvus Database
connections.connect(host=HOST, port=PORT)

utility.list_collections()
print("Connecting to Milvus",COLLECTION_NAME )
print("list of milvus collections", utility.list_collections())

# collection.is_empty
collection = Collection(COLLECTION_NAME)
collection.load()

print("Is Collection Empty?",collection.is_empty)

##################################################################
# Connect to the Milvus Vector Database (which is already running)
##################################################################




def query(query, top_k = 5):
    text,expr=query
    tmp1=hf.embed_documents([text])
    res = collection.search(tmp1, anns_field='embedding', expr = expr, param=QUERY_PARAM, limit = top_k, output_fields=['author', 'source_url', 'text'])
    
    context=[]
    references=[]
   
    for i, hit in enumerate(res):
        for ii, hits in enumerate(hit):
#             print('\t' + 'Rank:', ii + 1, 'Score:', hits.score, 'Author:', hits.entity.get('author'))
#             print('\t\t' + 'source_url:', hits.entity.get('source_url'))
#             print(textwrap.fill(hits.entity.get('text'), 1000))
#             print()
            wrapped_text = textwrap.fill(hits.entity.get('text'), 1000)
            context.extend([wrapped_text])
            wrapped_text = textwrap.fill(hits.entity.get('source_url'),1000)
            references.append([wrapped_text])
            
    return(context,references)


def answer_question(
    # context=context,
    # references=references,
    question, 
    model="gpt-3.5-turbo-0301",
  #  question=question,
 #   show_search_results=0,
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=300,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    # Get Context and references 
    print('question=',question)
    my_query=(question,"")
    context,references=query(my_query)
    
    # If debug, print the raw model response
    if debug:
        #print("Context:\n" + context)
        print("Context=",context)
        print("\n\n")

    try:
       
#### here is 3.5 turbo
        completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          temperature=0.8,
          max_tokens=max_tokens,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer the question. \nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-03-02"},
         # messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-03-02"},

          {"role": "user", "content" : "How are you?"},
          {"role": "assistant", "content" : "I am doing well"},
          {"role": "user", "content" : f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}]
        )
        url_list = [item[0] for item in references]
        answer_tmp=completion["choices"][0]["message"]['content'].strip()
        references_tmp=set(url_list)

        updated_urls = set()

        for url in references_tmp:
            updated_url = url.replace('borg.galaxiesoftware.com', 'bible.org')
            updated_urls.add(updated_url)

        references_tmp=updated_urls

        answer=f'{answer_tmp}{references_tmp}'

        answer=answer.replace('{', '<br><b>References:</b><br>').replace('}', '').replace('\'', "")


        print(answer)
        print("\nreferences:",*set(url_list),sep='\n')
        
       # print("context= ",context)
      #  return "gpt-3.5-turbo",completion["choices"][0]["message"]['content'].strip(),"Similar references",url_list
        #answer=completion["choices"][0]["message"]['content'].strip()

    except Exception as e:
        print(e)
        answer=e
        references_tmp={}
        #return ""
    return answer




def create_context(
    question, df, max_len=1800, size="ada"
):
    
    print("Question is: ", question)
    
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    # Get the embeddings for the question
    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_embeddings= model.encode(question).tolist()

    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def generateChatResponse(df,prompt):

    print(df.head())
    messages=[]
    messages.append({"role": "system", "content": "You are a helpful assistant."})
    
    max_len=1800
    size="ada"

    question={}
    question['role']='user'
    context=create_context(
        prompt,
        df,
        max_len=max_len,
        size=size,
    )
    question['content']=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {prompt}\nAnswer:"

    messages.append(question)

    print("messages=", messages)

    response= openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.8,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=messages)

    try: 
        answer=response['choices'][0]['message']['content'].replace('\n','<br>')
    except: 
        answer='Oops, something wrong with AI , try a different question, if problem persists come back later.'

    return answer


