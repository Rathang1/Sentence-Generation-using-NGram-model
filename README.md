# Sentence-Generation-using-NGram-model

## In this Natural Language processing project, I have created a N-Gram Language model to generate new sentences from the existing corpora.

### I calculate the MLE (Maximum Likelyhood Estimate) for a given word based on the previous N-words
### To implement that I created a vocabulary for an input text sample and stored the nGram counts along with the context counts
### The code also includes Laplace smoothing that prevents from getting an undefined value in case the context is not present in our dictionary
### Padding has been done with to mark the start and end of a tokenized sentence
### The code includes the calculation of entropy and perplexity as well and this is important as, if the input sentences get too long or the size for test data varies we can still have a standard value to compare the model performance, as well as it helps to solve the underflow problem that can happen as, if we multiply too many small probabilities for each n-gram then the value could underflow and taking the log helps in giving a larger absolute value that can be easily compared.
