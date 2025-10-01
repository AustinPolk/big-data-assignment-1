import os
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk # Use nltk 3.7
from pyspark import SparkContext, SparkConf

# Download the nltk necessity files
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")


def download_file() -> str:
    """
    Function download the sample book from gutenberg

    Returns:
        str: Name of the file downloaded file using wget
    """

    url = "https://www.gutenberg.org/ebooks/76952.txt.utf-8"
    fname = "76952.txt"

    # if file is not downloaded
    if not os.path.exists(fname):
        os.system(f"wget -O {fname} {url}")

    return fname


def extract_named_entities(text: str) -> list:
    """
    Function to extract named entities from the book's text

    Args:
        text (str): Text from the downloaded book

    Returns:
        list: List of named entities
    """

    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    chunks = ne_chunk(tags, binary=False)

    entities = []
    for subtree in chunks.subtrees():
        if subtree.label() != "S":  # skip root
            entity = " ".join(word for word, tag in subtree.leaves())
            entities.append(entity)

    return entities


if __name__ == "__main__":

    conf = SparkConf().setAppName("Gutenberg76952")
    sc = SparkContext(conf=conf)

    filename = download_file()

    # Load the book into RDD
    book_rdd = sc.textFile(filename)

    # Apply NER on each line of RDD parallely
    entities_rdd = book_rdd.flatMap(extract_named_entities)

    # print("Named entites: ")
    # print(entities_rdd.take(20))

    # Map -> Reduce -> Sort
    entity_counts = (
        entities_rdd
        .map(lambda e: (e,1))           # (named entity, 1)
        .reduceByKey(lambda a,b: a+b)   # sum counts
        .sortBy(lambda x: -x[1])        # sort by count (descending)
    )

    # Collect top 25 entities
    top_entities = entity_counts.take(25)

    print("\nTop 25 Named Entities in the book:")
    for entity, count in top_entities:
        print(f"{entity}: {count}")

    sc.stop()