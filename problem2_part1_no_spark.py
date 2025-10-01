import os
import pandas as pd
import nltk

nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math

def get_summaries():
    summaries_file = "plot_summaries.txt"

    if os.path.exists(f"./{summaries_file}"):
        return summaries_file

    source_url = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
    source_filename = "MovieSummaries"
    
    if not os.path.exists(f"./{source_filename}.tar.gz"):
        print(f"downloading tarball from {source_url}")
        os.system(f"wget {source_url}")

    if not os.path.exists(f"./{source_filename}"):
        print("unzipping")
        os.system(f"tar -xvzf {source_filename}.tar.gz")
    
    print("moving summaries file to current directory")
    os.system(f"mv {source_filename}/{summaries_file} ./{summaries_file}")

    assert os.path.exists(f"./{summaries_file}")
    print("complete, cleaning up")

    os.system(f"rm -rf {source_filename}")
    os.system(f"rm {source_filename}.tar.gz")

    return summaries_file

#def get_nltk_tools():
#    nltk.download("punkt")
#    nltk.download("stopwords")

stop_words = set(stopwords.words('english'))
def tokenize_and_remove_stop_words(document):
    tokens = word_tokenize(document)
    return [token for token in tokens if token not in stop_words]

if __name__ == "__main__":
    summaries_file = get_summaries()
    #get_nltk_tools()

    df = pd.read_csv(summaries_file, delimiter="\t", names=["movie_id", "plot_summary"])
    df["tokenized"] = df["plot_summary"].apply(tokenize_and_remove_stop_words)

    total_document_count = len(df.index)
    document_count_per_word = {}
    
    for _, row in df.iterrows():
        movie_id = row["movie_id"]
        plot_tokens = row["tokenized"]
        
        distinct_words = set(plot_tokens)
        for word in distinct_words:
            if word not in document_count_per_word:
                document_count_per_word[word] = 0
            document_count_per_word[word] += 1

    query_word = "apple"
    documents_with_query = document_count_per_word.get(query_word, 0)

    tf_idf_by_document = []
    for _, row in df.iterrows():
        movie_id = row["movie_id"]
        plot_tokens = row["tokenized"]

        total_words = len(plot_tokens)
        query_hits = plot_tokens.count(query_word)
        
        # calculate tf-idf
        tf = query_hits / total_words
        idf = 1 + math.log(total_document_count/documents_with_query)
        tf_idf_by_document.append((movie_id, tf/idf))

    sorted_results = sorted(tf_idf_by_document, key = lambda x: -x[1])
    print(sorted_results[:10])

