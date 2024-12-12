import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def sentence_to_random_embedding(sentence, embedding_dim=1024):
    # Tokenize the sentence using a simple bag-of-words method
    vectorizer = CountVectorizer()
    token_counts = vectorizer.fit_transform([sentence])
    vocabulary = vectorizer.get_feature_names_out()
    
    # Assign each word in the vocabulary a random 1024-dimensional vector
    word_embeddings = {word: np.random.randn(embedding_dim) for word in vocabulary}
    
    # Sum the embeddings of each word in the sentence
    sentence_embedding = np.zeros(embedding_dim)
    for word in vocabulary:
        sentence_embedding += token_counts[0, vectorizer.vocabulary_[word]] * word_embeddings[word]
    
    return sentence_embedding

# Example usage
sentence = "This is an example sentence to encode into a 1024-dimensional vector."
embedding = sentence_to_random_embedding(sentence)
print(embedding)
print("Embedding shape:", embedding.shape)
