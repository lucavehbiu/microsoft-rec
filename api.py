import sys
import pandas as pd
import numpy as np
import pickle
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def get_k_recommendations():
    # get parameters when calling a POST request
    args = request.args
    user_id = args.get('user_id')
    k = args.get('k')
    
    print('Example of List of Users to choose from:')
    print(model_input['_id'].head(10))
    print('\n')

    top_k_recommendations = recommender.recommend_top_k_items(model_input, k=int(k))
    similar_users = recommender.get_top_k_recommendations(model_input, user_id)['rec__id']
    return {f"{k} Most similar Users" : similar_users.to_json(orient = 'records')}


if __name__ == '__main__':
    try:
        recommender = pickle.load(open('model/tfidf_recommender.sav', 'rb'))
    except:
        import model.build_model
    # load model input data to predict on
    model_input = pd.read_csv('model_input/model_input.csv')
    
    app.run(debug=False, host="0.0.0.0", port=3001)


