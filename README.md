# Project description
The goal of this project is to enable a machine to **automatically classify a movie review** as either "positive" or "negative" based on a given dataset of review-sentiment pairs. The project uses a **supervised machine learning** paradigm.

<img src="https://github.com/andjadenic/Sentiment-Analysis/blob/main/readme_figure.png" alt="Alt text" width="30%">

## Dataset
"Large movie review dataset", also called "IMDB Review Dataset", is a dataset for binary sentiment classification containing 50,000 movie reviews labeled as `positive` or `negative`. This is a standard benchmark dataset for sentiment analysis. It’s widely used in NLP tasks.

* Length of review varies from 32 characters (min) to 13,704 characters (max) with a median length of 970 characters.  
* Reviews are from IMDB, allowing no more than 30 reviews per movie.
* The constructed dataset contains an even number of positive and negative reviews, 25,000 each.

> **Review example 1**: "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it. Great Camp!!!"
**Sentiment example 1**: "positive"


> **Review example 2**: "Encouraged by the positive comments about this film on here I was looking forward to watching this film. Bad mistake..."
**Sentiment example 2**: "negative"

Basic Exploratory Data Analysis is done in Jupiter notebook [eda.ipynb]().
Resources:
* ["Large movie review dataset" Maas, Andrew, et al. (2011)](https://ai.stanford.edu/~amaas/data/sentiment/)
* [Kaggle IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

### Data preprocessing and word embedding
Labels `negative` are coded to $0$, and labels `positive` are coded to $1$.

### Word embedding
Each review word is embedded using a pre-trained word2vec model that maps 3 million words into 300-dimensional vectors. The model is pre-trained on a portion of the Google News dataset (approximately 100 billion words) and downloaded using the Gensim API in [word_embeddin.py]().

Recourses:
* [Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space. (2013)"](https://arxiv.org/abs/1301.3781)
* [Rehurek, R., & Sojka, P. "Gensim–python framework for vector space modelling. (2011)](https://radimrehurek.com/gensim/intro.html)
* [Gensim pre-trained models](https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html)

After preprocessing and embedding, reviews are padded and, with corresponding labels, wrapped into torch's `Dataset` and `DataLoader` for training defined in [preprocessing/preprocessing.py]().

## Model architecture

The model is a multilayer LSTM with an additional fully connected linear layer and sigmoid activation function that produces a prediction.

1. **LSTM layer**: Processes the sequence of word embeddings, one at a time, capturing context and order of words.
2. **Fully connected (linear) layer**: Takes the final hidden state of the LSTM and maps it to a single value.
3. **Sigmoid activation**: Squashes that value into a probability between 0 and 1.

<img src="https://github.com/andjadenic/Sentiment-Analysis/blob/main/readme_figure_model_architecture.png" alt="Alt text" width="30%">

Resources:
* [Sak, Haşim, Andrew Senior, and Françoise Beaufays. "Long short-term memory based recurrent neural network architectures for large vocabulary speech recognition." (2014).](https://arxiv.org/abs/1402.1128)

## Training the Model
Training works by feeding batches of tokenized, padded, and embedded reviews into the model, one word at a time. The embeddings pass through the LSTM, then a linear layer, and finally a sigmoid to produce probabilities of positive sentiment. These predictions are compared with true labels using **binary cross-entropy loss**. The Adam optimizer updates model weights via backpropagation. This process repeats for several epochs until the model learns to classify reviews accurately.

$$
L = \sum_{n=1}^{N} y^{(n)} * \log(p^{(n)}) + (1 - y^{(n)}) * \log (1 - p^{(n)})
$$

where $N$ is size of training dataset, $y^{(n)}$ is label of $n$-th sample and $p^{(n)}$ is model's prediction for $n$-th sample.





