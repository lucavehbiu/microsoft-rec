import os
import sys
import pandas as pd
import numpy as np
import pickle
from recommenders.models.tfidf.tfidf_utils import TfidfRecommender


# Load Data
path = os.getcwd()
model_input_filename = 'model_input/model_input.csv'
data = pd.read_csv('data/recommender_tags.csv')

# Some pre-processing
data.replace('[]', np.NaN, inplace = True)
combined_tags = (data.communityTags.combine_first(data.interestTags)
                                   .combine_first(data.livingTags)
                                   .combine_first(data.needTags)
                                   .combine_first(data.offerTags)
                                   .combine_first(data.skillTags)
                )

combined_tags_df = (pd.concat([data._id, combined_tags], axis = 1)
                      .rename({'communityTags' : 'combined_tags'}, axis = 1)
)

combined_tags_df_na_dropped = combined_tags_df.dropna().reset_index(drop= True)

# Create the recommender object
recommender = TfidfRecommender(id_col='_id', tokenization_method='scibert')

# Clean tokens
model_input = recommender.clean_dataframe(combined_tags_df_na_dropped, 
                                          ['combined_tags'], 
                                          'cleaned_combined_tags'
                                         ).iloc[:500] # tfidf was taking too long, so i thought running with a sample
# save model_input for api.py
model_input.to_csv(model_input_filename, index = False)

# Tokenize text with tokenization_method specified in class instantiation
tf, vectors_tokenized = recommender.tokenize_text(model_input, text_col='cleaned_combined_tags')

# Fit the TF-IDF vectorizer
recommender.fit(tf, vectors_tokenized)

# save model after training
filename = f'{path}/model/tfidf_recommender.sav'
pickle.dump(recommender, open(filename, 'wb'))
