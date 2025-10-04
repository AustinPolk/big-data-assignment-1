import os
import nltk
import math
import json
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

    return [token for token in tokens if  token.isalpha() and token not in STOPWORDS]

def get_movie_names(sc, meta_path: str) -> dict:
    # Metadata: wiki_id -> movie_name
    meta_rdd = sc.textFile(meta_path)

    movie_name = (
        meta_rdd
        .map(lambda line: line.split("\t"))
        .filter(lambda parts: len(parts) > 2)
        .map(lambda parts: (parts[0], parts[2]))
        .collectAsMap()
    )

    b_movie_name = sc.broadcast(movie_name)
    return b_movie_name

def get_plot_tokens(sc, plot_path: str):
    # Plots: (wiki_id, summary)
    plots_rdd = (
        sc.textFile(plot_path)
        .map(lambda line: line.split("\t",1))
        .filter(lambda parts: len(parts)==2)
        .map(lambda parts: (parts[0], parts[1]))
    )

    # Tokenize the plots rdd
    tokens_rdd = plots_rdd.mapValues(tokenize_and_remove_stop_words)
    return tokens_rdd

def get_tfidf(tokens_rdd):
    # TF Term Frequencies: (wiki_id, word) -> count
    tf_rdd = (
        tokens_rdd
        .flatMapValues(lambda tokens: tokens)  # (wiki_id, word)
        .map(lambda x: ((x[0], x[1]), 1))      # ((wiki_id, word), 1)
        .reduceByKey(lambda a, b: a + b)       # ((wiki_id, word), count)
    )

    # TF Total Words: (wiki_id, length of summary tokens)
    summary_len_rdd = (
        tokens_rdd
        .map(lambda x: (x[0], len(x[1])))   # (wiki_id, summary_token_len) eg: (23890098, 5)
    )

    # Reshape the tf_rdd
    tf_by_doc_rdd = tf_rdd.map(lambda x: (x[0][0], (x[0][1], x[1])))  # (wiki_id, (word, count))

    tf_normalized_rdd = (
        tf_by_doc_rdd
        .join(summary_len_rdd)                                     # (wiki_id, ((word, count), summary_len)) eg: (23890098, ('hard-working',1), 5)
        .map(lambda x: ((x[0], x[1][0][0]), x[1][0][1]/x[1][1]))   # ((wiki_id, word), TF) eg. ((23890098, 'hard-working'), 0.2)
    )

    # -------------------------
    # LOGIC FOR IDF CALCULATION
    # -------------------------

    # DF: word -> number of documents containing it
    df_rdd = (
        tf_rdd
        .map(lambda x: (x[0][1], x[0][0]))   # (word, wiki_id)
        .distinct()
        .map(lambda x: (x[0], 1))            # (word, 1)
        .reduceByKey(lambda a, b: a + b)     # (word, df)
    )
    
    # Total number of documents
    N = tokens_rdd.count() 

    # IDF = log(total no. of docs / total no. of docs containting term)
    idf_rdd = df_rdd.mapValues(lambda df: math.log(N / df))

    # (word -> idf) as dictionary for broadcast
    idf_dict = dict(idf_rdd.collect())
    b_idf = sc.broadcast(idf_dict)

    # ----------------------------
    # LOGIC FOR TF-IDF CALCULATION
    # ----------------------------

    # Compute TF-IDF
    tfidf_rdd = tf_normalized_rdd.map(
        lambda x: (x[0], x[1] * b_idf.value.get(x[0][1], 0.0))  # ((wiki_id, word), tfidf)
    )

    return tfidf_rdd, b_idf

def process_single_term_query(query, tfidf_rdd, b_movie_name):
    # Filter tf-idf only for this query word
    results = (
        tfidf_rdd
        .filter(lambda x: x[0][1] == query)             
        .map(lambda x: (x[0][0], x[1]))                  # (wiki_id, tfidf)
        .sortBy(lambda x: -x[1])                         
        .take(10)                                        
    )

    print(f"\n\nTop 10 results for word: '{query}'\n")
    for wiki_id, score in results:
        mname = b_movie_name.value.get(wiki_id, "Unknown")
        print(f"\t{mname}: {score}")

def process_multi_term_query(query, b_idf, doc_tfidf_rdd, b_movie_name):
    # Tokenize the multi term queries
    q_tokens = tokenize_and_remove_stop_words(query)

    # Calculate query TF
    q_tf = {}
    for token in q_tokens:
        q_tf[token] = q_tf.get(token, 0) + 1                        # eg: q_tf = {"funny": 1, "movie": 1, "action": 1, "scenes": 1}
    q_len = len(q_tokens)
    q_tf = {word: count / q_len for word, count in q_tf.items()}    # eg: normalized q_tf = {"funny": 0.25, "movie": 0.25, "action": 0.25, "scenes": 0.25}

    # Calculate TF-IDF
    q_tfidf = {word: q_tf[word] * b_idf.value.get(word, 0.0) for word in q_tf}

    # Query norm
    q_norm = math.sqrt(sum(val * val for val in q_tfidf.values()))

    # Cosine similarity
    results_rdd = doc_tfidf_rdd.map(
        lambda x: (
            x[0],  # wiki_id
            sum(q_tfidf.get(word, 0.0) * x[1].get(word, 0.0) for word in q_tfidf) /
            (q_norm * math.sqrt(sum(val * val for val in x[1].values()))) if x[1] else 0.0
        )
    )

    results = results_rdd.sortBy(lambda x: -x[1]).take(10)

    print(f"\n\nTop 10 results for query: '{query}'\n")
    for wiki_id, score in results:
        mname = b_movie_name.value.get(wiki_id, "Unknown")
        print(f"\t{mname}: {score:.4f}")

if __name__ == "__main__":

    conf = SparkConf().setAppName("MovieSummaries_TFIDF")
    sc = SparkContext(conf=conf)

    # Download the tar.gz file
    download_file()

    # Choose plot file and metadata file containing the movie name
    plot_path = "MovieSummaries/plot_summaries.txt" if os.path.exists("MovieSummaries/plot_summaries.txt") else print("Error finding path")
    meta_path = "MovieSummaries/movie.metadata.tsv" if os.path.exists("MovieSummaries/movie.metadata.tsv") else print("Error finding path")

    # Get movie names from the metadata file
    b_movie_name = get_movie_names(sc, meta_path)

    # Tokenize each plot summary and store in an RDD
    tokens_rdd = get_plot_tokens(sc, plot_path)

    # Calculate tf-idf scores for each token
    tfidf_rdd, b_idf = get_tfidf(tokens_rdd)

     # Group document TF-IDF by document
    doc_tfidf_rdd = tfidf_rdd.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().mapValues(dict) 

    # === Process Search Queries ===

    # Get the queries from the search file
    with open("search.txt") as f:
        queries = [x.strip() for x in f.readlines()]

    # Process each query
    for query in queries:
        if ' ' not in query: # no spaces in the query indicates a single-word query
            process_single_term_query(query, tfidf_rdd, b_movie_name)
        else:
            process_multi_term_query(query, b_idf, doc_tfidf_rdd, b_movie_name)

    sc.stop()            

