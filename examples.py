import dat

# GloVe model from https://nlp.stanford.edu/projects/glove/
model = dat.Model("vectors_german.txt", "vocab_german.txt")

# Compound words are translated into words found in the model
print(model.validate("zahn")) # cul-de-sac

# Compute the cosine distance between 2 words (0 to 2)
print(model.distance("katze", "hund")) # 0.1983
print(model.distance("katze", "fingerhut")) # 0.8787

# Compute the DAT score between 2 words (average cosine distance * 100)
print(model.dat(["katze", "hund"], 2)) # 19.83
print(model.dat(["katze", "fingerhut"], 2)) # 87.87

# Word examples (Figure 1 in paper)
low = ["arm", "augen", "fuss", "hand", "kopf", "bein", "rhein"]
average = ["tasche", "biene", "burger", "fest", "hallo", "schuhe", "baum"]
high = ["nilpferd", "springer", "maschinen", "stachel", "tickets", "tomate", "violine"]

# Compute the DAT score (transformed average cosine distance of first 7 valid words)
print(model.dat(low)) # 50
print(model.dat(average)) # 78
print(model.dat(high)) # 95
