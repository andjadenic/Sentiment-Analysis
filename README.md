# Dataset
["Large movie review dataset" Maas, Andrew, et al. (2011)](https://ai.stanford.edu/~amaas/data/sentiment/), also called IMDB Review Dataset, is a dataset for binary sentiment classification containing 50,000 movie reviews labeled as `positive` or `negative`. This is a standard benchmark dataset for sentiment analysis. It’s widely used in NLP tasks.

* Length of review varies from 32 characters (min) to 13,704 characters (max) with a median length of 970 characters.  
* Reviews are from IMDB, allowing no more than 30 reviews per movie.
* The constructed dataset contains an even number of positive and negative reviews, 25,000 each.

**Review example 1**: "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it. Great Camp!!!"

**Sentiment example 1**: "positive"


**Review example 2**: "Encouraged by the positive comments about this film on here I was looking forward to watching this film. Bad mistake..."

**Sentiment example 2**: "negative"

Basic Exploratory Data Analysis is done in Jupiter notebook [eda.ipynb]().

## Data preprocessing and word embedding
Labels `negative` are coded to $0$, and labels `positive` are coded to $1$.

