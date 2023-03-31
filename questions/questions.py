import nltk
import sys
import os
import string
import math
FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    all_files = {}
    for f_name in os.listdir(directory):
        with open(os.path.join(directory, f_name), encoding="utf-8") as f:
            all_files[f_name] = f.read()
    return all_files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    words = nltk.tokenize.word_tokenize(document.lower())
    return [i for i in words if i not in string.punctuation and i not in nltk.corpus.stopwords.words("english")]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs ={} 
    words = set()
    for f_name in documents:
        words.update(documents[f_name])
    for word in words:
        freq =0
        for f_name in documents:
            if word in documents[f_name]:
                freq+=1
        idfs[word] = math.log(len(documents) / freq)
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = []
    for f in files:
        tfidf = 0
        for q in query:
            if q in files[f]:
                tfidf += files[f].count(q) * idfs[q]
        tf_idf.append((f, tfidf))
    tf_idf.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in tf_idf[:n]]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    res = []
    for sentence in sentences:
        freq_idf= sum(idfs[q] for q in query if q in sentences[sentence]) 
        freq_match =sum(q in query for q in sentences[sentence])
        density= float(freq_match)/len(sentences[sentence])
        res.append((sentence, freq_idf, density))
    res.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [x[0] for x in res[:n]]


if __name__ == "__main__":
    main()
