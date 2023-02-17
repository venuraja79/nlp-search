from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

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
print('Model loaded.')

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
  return render_template('index.html')


@app.route('/submit-search', methods=['POST'])
def perform_search():
  prompt = request.form['prompt-input']
  print(f"Generating search results for : {prompt}")

  # Get pinecone index
  index = pinecone.Index(index_name=INDEX_NAME)

  vector_embedding = model.encode(prompt).tolist()
  response = index.query([vector_embedding], top_k=3, include_metadata=True)

  search_result = {}
  if "matches" in response:
      results = []
      j_results = response['matches']
      for item in j_results:
          context = item['metadata']['context']
          title = item['metadata']['title']
          score = item['score']
          results.append({"context":context, "title":title, "score":score})

      search_result = {"result": results}
  else:
      search_result = {"result": "No response for this query! Please try a different one..."}

  print(f"Sending search results ... {response}")
  return render_template('index.html', generated_text=response)

if __name__ == '__main__':
    app.run()