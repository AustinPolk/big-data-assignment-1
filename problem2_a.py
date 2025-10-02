import os
import nltk
import math
from pyspark import SparkConf, SparkContext

# Download the nltk necessity files
nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))

def download_file():
    """
    Function download the tar.gz file from CMU Movie Summary Corpus

    """

    url = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
   
    # if file is not downloaded
    if not os.path.exists("MovieSummaries.tar.gz"):
        os.system(f"wget -O MovieSummaries.tar.gz {url}")
        os.system(f"tar -xvf MovieSummaries.tar.gz")


def tokenize_and_remove_stop_words(document: str) -> list:
    """
    Function to tokenize the document after removing the stopwords 

    Args:
        str of each document from the plots rdd 

    Returns:
        list of tokens after removal of stopwords
    """

    tokens = word_tokenize(document.lower())

    return [token for token in tokens if token not in STOPWORDS]


if __name__ == "__main__":

    conf = SparkConf().setAppName("MovieSummaries_TFIDF")
    sc = SparkContext(conf=conf)

    # Download the tar.gz file
    download_file()

    # Choose plot file and metadata file containing the movie name
    plot_path = "MovieSummaries/plot_summaries.txt" if os.path.exists("MovieSummaries/plot_summaries.txt") else print("Error finding path")
    meta_path = "MovieSummaries/movie.metadata.tsv" if os.path.exists("MovieSummaries/movie.metadata.tsv") else print("Error finding path")

    # Metadata: wiki_id -> movie_name
    meta_rdd = sc.textFile(meta_path)

    id_name = (
        meta_rdd
        .map(lambda line: line.split("\t"))
        .filter(lambda parts: len(parts) > 2)
        .map(lambda parts: (parts[0], parts[2]))
        .collectAsMap()
    )

    b_id_name = sc.broadcast(id_name)

    # Plots: (wiki_id, summary)
    plots_rdd = (
        sc.textFile(plot_path)
        .map(lambda line: line.split("\t",1))
        .filter(lambda parts: len(parts)==2)
        .map(lambda parts: (parts[0], parts[1]))
    )

    # Tokenize the plots rdd
    tokens_rdd = plots_rdd.mapValues(tokenize_and_remove_stop_words)

    # TF Term Frequencies: (wiki_id, word) -> count
    tf_rdd = (
        tokens_rdd
        .flatMapValues(lambda tokens: tokens)
        .map(lambda x: ((x[0], x[1]), 1))       # ((wiki_id, word) 1) eg: ((23890098, 'hard-working'), 1)
        .reduceByKey(lambda a, b: a+b)
    )
    
    # DF Document Frequencies: word -> set of documents
    df_rdd = (
        tf_rdd
        .map(lambda x: (x[0][1],x[0][0]))   # (word, wiki_id) eg: ('hard-working', 23890098)
        .distinct()
        .map(lambda x: (x[0], 1))              # (word, 1) eg: ('hard-working', 1)
        .reduceByKey(lambda a, b: a+b)
    )

    # Total number of documents
    N = tokens_rdd.count() 

    # IDF = log(total no. of docs / total no. of docs containting term)
    idf_rdd = df_rdd.mapValues(lambda df: math.log(N / df)) 

    # (word -> idf) as dictionary for broadcast
    idf_dict = dict(idf_rdd.collect())
    b_idf = sc.broadcast(idf_dict)

    # Calculate TF-IDF
    tfidf_rdd = tf_rdd.map(lambda x: (x[0], x[1] * b_idf.value.get(x[0][1], 0.0)))
    


    

