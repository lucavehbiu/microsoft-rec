# Recommending System using Microsoft recommenders

TF-IDF model was used to create a recommender. It was the only option available from Microsoft recommenders
that dealt with text data. Recommenders usually work on a user-product-rating framework and our
dataset is not build like that. It has users and the tags they use.

Maybe, some further investigation can show that other models can be used in this case.

TF-IDF is a standard model used for text data. It basically tokenizes all text, in a standard
format and builds the frequency of each token. Building from that it calculates cosine
similarity of the vector tokens of a user with another one and recommends accordingly.

Cosine similarity is a distance metric used for vector tokens because of its nature of high-dimensionality.
Euclidean distance does not work in this cases and basically cosine distance measures the angle between
different vector of tokens.

# Obstacles
TF-IDF does not scale well so I took a small sample of 500 users to build a model. In my machine it crashed
when I tried to do all users at once.

# To train a new model
- delete model in /model/ and it will create a new model for the new input you have provided

# Run to create image
- docker build --tag  microsoft-recommender .

# Run to start up container from created image
- docker run -p 3001:3001 --name backend microsoft-recommender
