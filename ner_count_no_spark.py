import nltk

# read lines from document
with open("pg76952.txt", encoding="utf-8") as f:
    content_lines = [x.strip() for x in f.readlines()]

# download necessary NLTK tools
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")

# helper for getting all named entities from a line in a single list
def get_named_entities(line):
    tokens = nltk.word_tokenize(line)
    tagged_tokens = nltk.pos_tag(tokens)
    named_entities = nltk.ne_chunk(tagged_tokens)
    
    entities = []
    for entity in named_entities:
        if isinstance(entity[0], tuple):
            entities.append(' '.join((x[0] for x in entity)))

    return entities

# get a flat list of all named entities in the whole document
named_entities_per_line = [get_named_entities(line) for line in content_lines]
named_entities = []
for per_line in named_entities_per_line:
    for named_entity in per_line:
        named_entities.append(named_entity)

# count the number of occurrences of each entity
named_entities_count = {}
for named_entity in named_entities:
    if named_entity not in named_entities_count:
        named_entities_count[named_entity] = 0
    named_entities_count[named_entity] += 1

# sort descending to see which entities had the highest counts
counts = [(x, y) for x, y in named_entities_count.items()]
sorted_counts = sorted(counts, key = lambda x: -x[1])
print(sorted_counts)
