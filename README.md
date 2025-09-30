# Prerequisites:
  - WSL/Linux
  - PySpark

# Development
PySpark code is written like normal Python code in files ending in ".py". Remember to create the SparkContext manually, since we aren't running through the PySpark command line anymore.

# Testing
Run "spark-submit <path-to-source-code>" to run the code you've written. This will create a "results" folder in the same directory as the source code.

# Files
  - *wordcount.py* is a simple example of getting the word count for distinct words in a document
  - *pg76952.txt* a sample document from the Gutenberg project to use for Problem 1.
