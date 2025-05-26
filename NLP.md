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

* Word Embeddings (Word2Vec, GloVe)
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
