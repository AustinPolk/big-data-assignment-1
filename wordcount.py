from pyspark import SparkContext

sc = SparkContext.getOrCreate()
# spark = SparkSession.builder.appName('WordCount').getOrCreate()

content = sc.textFile("pg76952.txt")
words = content.flatMap(lambda x: x.split(" ")).map(lambda x: x.lower())
word_counts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x+y)
word_counts.saveAsTextFile("results")

