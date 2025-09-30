# Dataset
["Large movie review dataset" Maas, Andrew, et al. (2011)](https://ai.stanford.edu/~amaas/data/sentiment/), also called IMDB Review Dataset, is a dataset for binary sentiment classification containing 50,000 movie reviews labeled as `positive` or `negative`. This is a standard benchmark dataset for sentiment analysis. It’s widely used in NLP tasks.

* Length of review varies from 32 characters (min) to 13,704 characters (max) with a median length of 970 characters.  
* Reviews are from IMDB, allowing no more than 30 reviews per movie.
* The constructed dataset contains an even number of positive and negative reviews, 25,000 each.

> **Review example 1**: "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it. Great Camp!!!"

> **Sentiment example 1**: "positive"


**Review example 2**: "Encouraged by the positive comments about this film on here I was looking forward to watching this film. Bad mistake..."

**Sentiment example 2**: "negative"

Basic Exploratory Data Analysis is done in Jupiter notebook [eda.ipynb]().

## Data preprocessing and word embedding
Labels `negative` are coded to $0$, and labels `positive` are coded to $1$.

### Word embedding
Each review word is embedded using a pre-trained word2vec model ( [Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space. (2013)"](https://arxiv.org/abs/1301.3781) )

Pre-trained vectors are trained on a part of the Google News dataset (about 100 billion words) and downloaded using Gensim API ( [Rehurek, R., & Sojka, P. "Gensim–python framework for vector space modelling. (2011)](https://radimrehurek.com/gensim/intro.html) ) in [word_embeddin.py]().
* Pre-trained word2vec model maps 3 million words into 300-dimensional vectors.




