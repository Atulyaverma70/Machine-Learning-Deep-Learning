## Natural language processing
NLP is a field of AI that focuses on the intraction between human and computers.

* Tokenization
    * It is the technique used to split the text into words.
    * eg- I love NLP.-> 'I','love', 'NLP'

### Lemmatization and Stemming
    * Both are used to convert the words into base forms.
* Stemming
    * Removes the suffix to bring a word into its root form.
    * eg: running-> runn, flies->fli
* Lemmatization
    * redue the words into root form.
    * eg: running->run, flies->fly
* Part of Speech
    * Helps in understanding the context and meaning of a sentence.
* Named Entity recognition
    * NER identifies and classifies entities in a text into predefined categories:
    * eg: 
        * Person – "Elon Musk"
        * Organization – "Tesla"
        * Location – "California"
        * Date/Time – "January 2024"
        * Product – "iPhone"

* Syntactic Analysis (Parsing)
    * Analyzes the structure of a sentence (grammar).
    * Focuses on how words are organized.
    * Uses dependency parsing and constituency parsing.
    * Example:"I went to the bank."
        * Bank → Financial institution OR Riverbank → Semantic analysis resolves this.



* Semantic Analysis
    * Analyzes the meaning of a sentence.
    * Deals with ambiguity and contextual understanding.
    * Uses techniques like word sense disambiguation and semantic role labeling.
    * Example:"I went to the bank."
        * Bank → Financial institution OR Riverbank → Semantic analysis resolves this.

* Word Embeddings (Word2Vec, GloVe)- its a tool that captures semantic similarity.
    * Word embeddings are vector representations of words that capture their meanings and relationships.
        * Word2Vec
            * Predicts the context of a word using a neural network.
* language models like BERT, GPT are used to prdict next word.

## Vectorization techniques(used to convert words in vector(numbers) form)
* One hot Encoding 
* Bag of words
* word embedding(represent the average meaning of words into vectors)
    *  they are static(once you declared then i didnt changed)
* Contextual embedding(it is dynamic)


## full NLP in structured way
### What?
### why
### Components of NLP
there different levels of language processing
1. Morphological Analysis
    * Dels with structure of words
    * focuses on morphemes: smallest unit of meaning(eg un, break, able in unbreakable)
2. Syntactic analysis
    * Detemine the grammar and relationships between words using rules.
    * Techniques: dependency parsing
    * tools: spaCy, NLTK
3. semantic analysis
    * Assigns meaning to syntactically valid sentences.
    * eg Understanding that "apple" in "He ate an apple" refers to fruit, not the company.
    * Techniques: NER, Word embedding(Word2Vec,BERT)
4. Discourse Integration
    * Maintains the coherence across sentence.
    * sen1: Ravi dropped the glass.
    * sen2: It brokes.
        * here It refers to glass, not ravi.
    * Techniques:AllenNLP, spCy
5. Progmatic Analysis:
    * Understand the intent and context behind sentence.
    * eg. "can you pass the salt?" is not a yes/no question- its request.
    * Techniques: BERT 

### Text Preprocessing techniques

1. Tokeniztion
2. Lowecasing
3. Stopword Removal
4. Stemming
5. Lemmatization
6. POS tagging
7. Regualar Expressions- used to remove all the symbols, numbers, the

### Text representation techniques
1. One hot encoding 
    * reprsent each word in binary vector of length equal to the vocabulary size. Only the position of the word is 1, rest are 0.
    * How: given vocab:["I","like","dogs"]
        * "I": [1,0,0]
        * "like":[0,1,0]
        * "dogs":[0,0,1]
    * high dimensionality for large vocabularies
2. Bag of Words
    * reprrsents a docs by the frequency of each word in vocab.
    * simple efficient of many ML tasks like spam detection, sentiment analysis.
    * How
        * doc1: "I love dogs"
        * doc2: "I hate cats"
            * Vocab: [I, love, dogs, hate, cats]
            * BOW of doc1-[1,1,1,0,0],  doc2-[1,0,0,1,1]
    * dosent capture context or semantic
3. TF-IDF
    * Improves BoW by giving weights to words based on their importance. Rare words in the corpus get higher weight, frequent ones (like "the", "is") get lower.
    * why : To reduce the impact of common but less informative words and highlight important terms.
    *  How:
        * TF = Frequency of term in a document
        * IDF = log(Total documents / Number of documents with the term)
        * TF-IDF = TF × IDF
            * why IDF
                ```
                The Goal of TF-IDF:
                We want to highlight words that are important in a specific document but not too common in all documents.

                🧠 Why "Inverse Document Frequency"?
                Imagine two words:

                "data" appears in 2 out of 100 documents

                "the" appears in 98 out of 100 documents

                Now, if we just used TF (Term Frequency), both could get high scores in their respective documents.
                But "the" is common in almost every document, so it's not helpful in telling one document from another.

                So, we introduce IDF:

                text
                Copy
                Edit
                IDF = log(Total Documents / Docs with the term)
                This gives:

                IDF("data") = log(100 / 2) = high → important term

                IDF("the") = log(100 / 98) = low → not important

                By taking the inverse, we penalize words that appear in many documents.
    * Cannot capture contextual meaning

4. Count vectorization
    ```
    ✅ What:
    Like BoW, it converts a collection of text documents into a matrix of token counts.

    ❓ Why:
    A simple and fast way to prepare text data for ML models.

    ⚙️ How:
    Scikit-learn’s CountVectorizer builds vocabulary and encodes each document based on word counts.

    Example:
    Sentences:

    "I love NLP"

    "NLP is fun"

    Vocabulary: ['I', 'love', 'NLP', 'is', 'fun']

    Matrix:

    csharp
    Copy
    Edit
    [1, 1, 1, 0, 0]
    [0, 0, 1, 1, 1]
    ⚠️ Limitations:
    Same as BoW

    Cannot distinguish between “NLP is fun” and “fun is NLP”

5. Word Embedding
    * what and why
    ```
    1. Word embeddings are a way to represent words as dense numerical vectors in a continuous vector space where semantic relationships between words are preserved.
    2. Instead of high-dimensional sparse vectors (like in BoW or one-hot), embeddings use low-dimensional dense vectors (e.g., 100–300 dimensions).
     Why do we use Word Embeddings?

    Because they solve key problems in traditional methods like:
    No semantic meaning in BoW / TF-IDF
    Large and sparse vector sizes
    Can't capture context, similarity, or relationships
    Embeddings help:
    Group similar words (king, queen, emperor)
    Enable mathematical operations on word meanings
    Reduce dimensionality

    ```
    * How it work: They are learn ed using unsupervised model on large text corpora.
    * Key Models for Learning Embeddings:
        ```
        🔸 1. Word2Vec (Google)
        ✅ Techniques:
        CBOW (Continuous Bag of Words):

        Predict the current word based on the context

        Example: "I ___ to school" → Predict "go"

        Skip-Gram:

        Predict surrounding words given the current word

        Example: "go" → Predict: "I", "to", "school"

        🔄 Training:
        Trains a shallow neural network

        Learns vector representations that reduce prediction loss

        🔍 Example:
        Vector("king") - Vector("man") + Vector("woman") ≈ Vector("queen")

        🔸 2. GloVe (Global Vectors for Word Representation) – Stanford
        ✅ Concept:
        Uses a word-word co-occurrence matrix from the entire corpus

        Learns embeddings that best reconstruct the co-occurrence stats

        🔧 Difference from Word2Vec:
        Word2Vec is local (focuses on a small context window)

        GloVe is global (considers how often words appear together overall)

        🔸 3. FastText (Facebook)
        ✅ Enhancement:
        Learns subword (n-gram) embeddings (e.g., "play", "playing", "played")

        Word vector = sum of its character n-grams

        🚀 Benefit:
        Handles out-of-vocabulary (OOV) words like typos, rare words

        Understands morphology: "walking" and "walked" are similar

### Language modeling
* A language model is a statistical or neural model that assigns a probability to a sequence of words
     * why assigning the probability
     ```
     This helps us quantify how natural or grammatically correct a sentence is based on what it has learned from language data.

    🧠 Example:
    "I love NLP" → High probability

    "Love I NLP" → Low probability

    The second sentence is syntactically incorrect. The model learns this from data and assigns it a lower probability.
    Summary:
    A language model assigns probabilities to word sequences to understand, predict, and generate human-like language. This numerical probability is the key to making machines “understand” how language works.
1. N-gram Language Models
    * Predicts the next word using the previous (n−1) words:
2. Smoothing Techniques
    *  Smoothing is a technique used to handle zero probabilities in language models, especially N-gram models, when some word sequences are not seen in the training data.
    * Example
        ```
        🧠 Example:
        Corpus:

        "I love NLP"

        "You love AI"

        Now ask:

        What is P("I love AI")?

        "I love" exists → OK

        "love AI" exists → OK

        "AI" following "I"? Never seen → Zero probability

        If one component is zero → entire sequence becomes zero.
3. Perplexity
    * A metric to evaluate how good a language model is.
    * Intuition:
    ```
    Perplexity tells us "how surprised" the model is by the sequence.

    A perfect model (predicts correctly every time) has PP = 1

    A worse model has PP → ∞

###  Sequence Models
1. Recurrent Neural Networks (RNN)
2. Long Short-Term Memory (LSTM)
3. Gated Recurrent Units (GRU)
4. Attention Mechanism
5. Bidirectional RNNs/LSTMs
6. Encoder-Decoder architecture

###  Transformers and BERT-based Models
1. Transformers (self-attention, multi-head attention, positional encoding)
2. BERT (Bidirectional Encoder Representations from Transformers)
3. GPT (Generative Pre-trained Transformer)
4. RoBERTa, DistilBERT, XLNet, T5
5. Fine-tuning vs Feature extraction
6. Transfer Learning in NLP

### Sequence Labeling
Sequence labeling is a type of supervised learning where each element in an input sequence (typically words in a sentence) is assigned a label.
* example
    ```
    Example:
    Sentence: "Barack Obama was born in Hawaii."
    Labels:

    Barack → B-PER

    Obama → I-PER

    was → O

    born → O

    in → O

    Hawaii → B-LOC

1.  Named Entity Recognition (NER)
    ```
    Identify and classify named entities in text into predefined categories such as:

    PER – Person

    LOC – Location

    ORG – Organization

    MISC – Miscellaneous

    ✔️ Example:
    "Apple Inc. is based in California."
    → Apple Inc. → B-ORG I-ORG
    → California → B-LOC

    ✔️ Use Cases:
    Information extraction

    Question answering systems

    Knowledge base population

2. Part-of-Speech (POS) Tagging
    ```
    ✔️ Purpose:
    Assign each word in a sentence a grammatical category like:

    Noun (NN)

    Verb (VB)

    Adjective (JJ)

    Preposition (IN)

    Determiner (DT)

    ✔️ Example:
    "She eats apples."
    → She → PRP
    → eats → VBZ
    → apples → NNS

    ✔️ Use Cases:
    Syntactic parsing

    Grammar checking

    Linguistic analysis

