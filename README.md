# Article-Headline-Prediction-using-Attention-Model

The Problem statement given is to effectively predict the headline or a title of a given newsor a social media post by identifying the set of keywords.
Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding.
Text summarization is a method in natural language processing (NLP) for generating a short and precise summary of a reference document. Producing a summary of a large document manually is a very difficult task. 
To summarize text effectively, deep learning models need to be able to understand documents and discern and distill the important information. 

# Recurrent Neural Networks
Recurrent neural networks have recently been found to be very effective for many transduction tasks
- that is transforming text from one form to another. Examples of such applications include machine
translation and speech recognition. These models are trained on large amounts of input and
expected output sequences, and are then able to generate output sequences given inputs never before
presented to the model during training.
Recurrent neural networks have also been applied recently to reading comprehension. There, the
models are trained to recall facts or statements from input text.
Recurrent neural networks have recently been found to be very effective for many transduction tasks
- that is transforming text from one form to another. Examples of such applications include machine
translation and speech recognition. These models are trained on large amounts of input and
expected output sequences, and are then able to generate output sequences given inputs never before
presented to the model during training.
Recurrent neural networks have also been applied recently to reading comprehension. There, the
models are trained to recall facts or statements from input text.


# Encoder Decoder Architecture
The encoder-decoder architecture for recurrent neural networks is proving to be powerful on a host of sequence-to-sequence prediction problems in the field of natural language processing such as machine translation and caption generation. The encoder is fed as input the text of a news article one word of a time. Each word is first passed
through an embedding layer that transforms the word into a distributed representation. That distributed representation is then combined using a multi-layer neural network with the hidden layers
generated after feeding in the previous word, or all 0’s for the first word in the text.
The decoder takes as input the hidden layers generated after feeding in the last word of the input text.
First, an end-of-sequence symbol is fed in as input, again using an embedding layer to transform the
symbol into a distributed representation. Then, the decoder generates, using a softmax layer and the
attention mechanism, described in the next section, each of the words of the headline, ending with
an end-of-sequence symbol. After generating each word that same word is fed in as input when
generating the next word.


# Attention Layer
Attention is a mechanism that addresses a limitation of the encoder-decoder architecture on long sequences, and that in general speeds up the learning and lifts the skill of the model no sequence to sequence prediction problems. Attention is a mechanism that helps the network remember certain aspects of the input better, including names and numbers. The attention mechanism is used when outputting each word in the
decoder. For each output word the attention mechanism computes a weight over each of the input
words that determines how much attention should be paid to that input word. The weights sum up to
1, and are used to compute a weighted average of the last hidden layers generated after processing
each of the input words. This weighted average, referred to as the context, is then input into the
softmax layer along with the last hidden layer from the current step of the decoding.

# LSTM units
LSTMs help preserve the error that can be backpropagated through time and layers. By maintaining a more constant error, they allow recurrent nets to continue to learn over many time steps (over 1000), thereby opening a channel to link causes and effects remotely.
Those gates act on the signals they receive, and similar to the neural network’s nodes, they block or pass on information based on its strength and import, which they filter with their own sets of weights. Those weights, like the weights that modulate input and hidden states, are adjusted via the recurrent networks learning process. That is, the cells learn when to allow data to enter, leave or be deleted through the iterative process of making guesses, backpropagating error, and adjusting weights via gradient descent.


# References
1. Konstantin Lopyrev "Generating News Headlines with Recurrent Neural Networks"
2. https://skymind.ai/wiki/lstm
3. https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
4. Jon Krohn lectures on Natural Language Processing
5. https://machinelearningmastery.com/encoder-decoder-deep-learning-models-text-summarization/
6. https://www.sas.com/en_in/insights/analytics/what-is-natural-language-processing-nlp.html
7. http://home.iitk.ac.in/~soumye/cs498a/pres.pdf




