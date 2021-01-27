import pickle
import numpy as np, pandas as pd 
from flask import Flask, render_template, request
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
import en_core_web_md
nlp = en_core_web_md.load() 

app = Flask(__name__)
# pickle.dump(recommend, open('model.pkl','wb'))
model = pickle.load(open('../Deployment/model/model_rfw2v.pkl','rb'))

train = pd.read_csv('../datasets/final combined file/troll_data.csv')
# list_sequences_train = train["content"]
# max_features = 22000
# tokenizer = Tokenizer(num_words=max_features)
# train = tokenizer.fit_on_texts(list(list_sequences_train))

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def get_vec(x):
  doc = nlp(x)
  return doc.vector

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        comment = [x for x in request.form.values()]
        comment = [preprocess_text(comment[0])]
        comment = pd.DataFrame(comment)
        comment = comment[0].apply(lambda x: get_vec(x))
        XX = comment.to_numpy()
        XX = XX.reshape(-1,1)
        XX = np.concatenate(np.concatenate(XX,axis = 0),axis = 0).reshape(-1,300)
        data=XX
        prediction = model.predict(data)
        return render_template('index.html', predict = prediction)

if __name__ == '__main__':
    app.run(debug=True)