from flask import Flask, render_template, jsonify, request
import config
import openai
import aiapi

import pandas as pd
import numpy as np


def page_not_found(e):
  return render_template('404.html'), 404


app = Flask(__name__)
app.config.from_object(config.config['development'])

app.register_error_handler(404, page_not_found)


# print("reading embeddings for bible.org into dataframe")    
# df2=pd.read_csv('processed/borg_embeddings.csv', index_col=0)
# df2['embeddings'] = df2['embeddings'].replace(r'\n',' ', regex=True)
# df2['embeddings'] = df2['embeddings'].replace(r'   ',',', regex=True)
# df2['embeddings'] = df2['embeddings'].replace(r'  ',',', regex=True)
# df2['embeddings'] = df2['embeddings'].replace(r' ',',', regex=True)
# df2['embeddings']=df2['embeddings'].tolist()

# print("processing embedddings")
# df3=df2
# embeddings_list = []

# for i, row in df2.iterrows():
#                  #print(i,row['embeddings'])
#                  if row['embeddings'][1] == ',':
#                     #print("comma detected")
#                     s=row['embeddings'][0]+row['embeddings'][2:]
#                  else: 
#                     s=row['embeddings']
#                  embeddings_list.append([float(x.strip(' []')) for x in s.split(',')])
                 
# df3['embeddings']=embeddings_list


# ##################################################################
# # Connect to the Milvus Vector Database (which is already running)
# ##################################################################
# HOST = 'localhost'
# PORT = 19530
# COLLECTION_NAME = 'borg_data'
# DIMENSION = 768
# #DIMENSION = 1536
# #OPENAI_ENGINE = 'text-embedding-ada-002'

# openai.api_key = "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"

# INDEX_PARAM = {
#     'metric_type':'L2',
#     'index_type':"HNSW",
#     'params':{'M': 8, 'efConstruction': 64}
# }

# QUERY_PARAM = {
#     "metric_type": "L2",
#     "params": {"ef": 64},
# }

# BATCH_SIZE = 1000

# from pymilvus import connections, utility, FieldSchema, Collection, CollectionSchema, DataType

# # Connect to Milvus Database
# connections.connect(host=HOST, port=PORT)

# utility.list_collections()
# print("Connecting to Milvus",COLLECTION_NAME )
# print("list of milvus collections", utility.list_collections())

# # collection.is_empty
# collection = Collection(COLLECTION_NAME)
# collection.load()

# print("Is Collection Empty?",collection.is_empty)

# ##################################################################
# # Connect to the Milvus Vector Database (which is already running)
# ##################################################################


#print(df3.head())

print("starting app")

@app.route('/', methods = ['POST', 'GET'])

def index():

    if request.method == 'POST':
        question=request.form['prompt']

        res={}
      #  res['answer']=aiapi.generateChatResponse(df3,prompt)
        res['answer']=aiapi.answer_question(question)
        return jsonify(res),200

    return render_template('index.html', **locals())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888', debug=True)
