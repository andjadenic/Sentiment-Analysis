# Dataset
["Large movie review dataset" Maas, Andrew, et al. (2011)](https://ai.stanford.edu/~amaas/data/sentiment/), also called IMDB Review Dataset, is a dataset for binary sentiment classification containing 25,000 movie reviews for training, and 25,000 for testing, labeled as positive or negative. This is a standard benchmark dataset for sentiment analysis. It’s widely used in NLP tasks.

* Reviews are from IMDB, allowing no more than 30 reviews per movie.
* The constructed dataset contains an even number of positive and negative reviews.

Sentence example:
Class example:

### Preprocessing

IMDB review dataset is integrated into `torchtext` using the `torchdata` datapipe API. So, instead of manually downloading and preprocessing, create a datapipe that:
* Downloads the dataset.
* Extracts the files.
* Parses each review and its label.
* Outputs an iterable stream of (text, label) samples.

