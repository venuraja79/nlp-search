from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

import os
import torch
import pinecone
from sentence_transformers import SentenceTransformer

INDEX_NAME, INDEX_DIMENSION = 'squad', 384
MODEL_NAME = 'sentence-transformers/msmarco-MiniLM-L6-cos-v5'
PINECONE_KEY = 'pinecone_key'

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Loading model from _Sentence Transformers_: `{MODEL_NAME}` from Sentence Transformers to `{device}`...')
model = SentenceTransformer(MODEL_NAME, device=device)
print(f'Loaded the Model {MODEL_NAME}')

from transformers import pipeline
device = 0 if torch.cuda.is_available() else -1
model_name = "deepset/electra-base-squad2"
# load the reader model into a question-answering pipeline
reader = pipeline(tokenizer=model_name, model=model_name, task="question-answering", device=device)
print(f'Loaded reader pipeline model {model_name}')

from transformers import BartTokenizer, BartForConditionalGeneration

gen_model = "vblagoje/bart_lfqa"
# load bart tokenizer and model from huggingface
tokenizer = BartTokenizer.from_pretrained(gen_model).to(device)
generator = BartForConditionalGeneration.from_pretrained(gen_model).to(device)

print(f'Loaded generator pipeline model {gen_model}')

pinecone_key = os.getenv(PINECONE_KEY)

if os.getenv("index_name"):
    INDEX_NAME = os.getenv("index_name")

pinecone.init(
    api_key=pinecone_key,
    environment='us-east1-gcp'  # find in console next to api key
)

# Start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def initial():
  search_result = {"results":'', "qa_results":'', "lfqa_result": ''}
  search_result['vars'] = {}
  search_result['vars']['search_query'] = ''
  search_result['vars']['gen_qa'] = 'no'
  return render_template('index.html', **search_result)

def qa_pipeline(question, context):
    results = []
    for c in context:
        # feed the reader the question and contexts to extract answers
        answer = reader(question=question, context=c)
        # add the context to answer dict for printing both together
        start = int(answer['start'])
        end = int(answer['end'])
        answer["context"] = c[:start] + " <mark> " + c[start:end] + " </mark> " + c[end:]
        results.append(answer)
    # sort the result based on the score from reader model
    sorted_result = sorted(results, key=lambda x: x["score"], reverse=True)
    return sorted_result[0:1]

def lfqa_pipeline(question, context):
    context = [f"<P> {m['metadata']['context']}" for m in context]
    # concatinate all context passages
    context = " ".join(context)
    # contcatinate the query and context passages
    query = f"question: {question} context: {context}"

    # tokenize the query to get input_ids
    inputs = tokenizer([query], max_length=1024, return_tensors="pt")
    # use generator to predict output ids
    ids = generator.generate(inputs["input_ids"], num_beams=12, min_length=20, max_length=200, temperature=0.8)
    # use tokenizer to decode the output ids
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer

@app.route('/submit-search', methods=['get'])
def perform_search():
  s_query = request.args.get('query',' ')
  s_gen_qa = request.args.get('gen_qa', 'no')
  print(f"Generating search results for : {s_query}")

  # Get pinecone index
  index = pinecone.Index(index_name=INDEX_NAME)

  vector_embedding = model.encode(s_query).tolist()
  response = index.query([vector_embedding], top_k=5, include_metadata=True)

  search_result = {}
  if "matches" in response:
      results = []
      j_results = response['matches']
      for item in j_results:
          context = item['metadata']['context']
          title = item['metadata']['title']
          score = item['score']
          results.append({"context":context, "title":title, "score":score})

      # QA pipeline
      context = [x["metadata"]["context"] for x in response["matches"]]
      qa_result = qa_pipeline(s_query, context[:3])

      if s_gen_qa == 'yes':
          lfqa_context = response['matches']
          gen_answer = lfqa_pipeline(s_query, lfqa_context)
      else:
          gen_answer = ''

      search_result = {"results": results, "qa_results": qa_result, "lfqa_result": {"gen_content": gen_answer}}

  search_result['vars'] = {}
  search_result['vars']['search_query'] = s_query
  search_result['vars']['gen_qa'] = s_gen_qa
  print(f"Sending search results ... {search_result}")
  return render_template('index.html', **search_result)

if __name__ == '__main__':
    app.run()