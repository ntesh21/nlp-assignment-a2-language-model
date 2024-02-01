Task 1
-------------------------------------------

### Load Data

Load Harry Potter Corpus
Source: https://github.com/ErikaJacobs/Harry-Potter-Text-Mining/tree/master

##### About Dataset

The dataset is taken from the github repi of **ErikaJacobs**. In this repo ErikaJacobs has performed a **Text Analysis of the Harry Potter Book Series**. 
 - "This project features sentiment analysis conducted on the text of the Harry Potter book series by JK Rowling."

This repository contains the Harry Potter dataset under the folder **Book Text**. The folder contains 7 txt files each containing the text from single chapter or books. The files are:
   * HPBook1.txt
   * HPBook2.txt
   * HPBook3.txt
   * HPBook4.txt
   * HPBook5.txt
   * HPBook6.txt
   * HPBook7.txt

These text files were last committed four years ago. The size of the files are varying from 450kb to 1.4mb depending on the content of each book.

According to the repo, the source for this file is another github repo **bradleyboehmke** : https://github.com/bradleyboehmke/harrypotter

The details for the data in this repo is mentioned as:
------------------------------------------------------

An R Package for J.K. Rowling's Harry Potter Series

This package provides access to the full texts of the first seven Harry Potter books. The UTF-8 plain text for each novel was sourced from [Read Vampire Books](www.readbooksvampire.com) **however the website is not currently accessible**, processed a bit, and is ready for text analysis. Each text is in a character vector with each element representing a single chapter. The package contains:

-   `philosophers_stone`: Harry Potter and the Philosophers Stone, published in 1997
-   `chamber_of_secrets`: Harry Potter and the Chamber of Secrets, published in 1998
-   `prisoner_of_azkaban`: Harry Potter and the Prisoner of Azkaban, published in 1999
-   `goblet_of_fire`: Harry Potter and the Goblet of Fire, published in 2000
-   `order_of_the_phoenix`: Harry Potter and the Order of the Phoenix, published in 2003
-   `half_blood_prince`: Harry Potter and the Half-Blood Prince, published in 2005
-   `deathly_hallows`: Harry Potter and the Deathly Hallows, published in 2007

* For this assignment we will use 
   - (HPBook1.txt HPBook2.txt HPBook3.txt HPBook4.txt) as training data
   - (HPBook5.txt HPBook6.txt) as validation data
   - (HPBook7.txt) as test data


Task 2
---------------------------------------------

### Preprocessing Steps

##### **Tokenization**: 
* The **get_tokenizer** function from torchtext is used to create a basic English tokenizer. This tokenizer breaks down text into individual words.
* *lambda* function is defined to apply tokenization to each example in the dataset. The resulting tokens are stored in a new field named **tokens**.
* The *map* function is used to apply this tokenization function to each example in the dataset, and the original text column is removed.

##### **Numericalizing**:
* The vocabulary (vocab) is built from the tokenized training dataset. It assigns a unique numerical index to each token that appears at least three times (min_freq=3).
* Two special tokens, <unk> (unknown) and <eos> (end of sequence), are inserted into the vocabulary at indices 0 and 1, respectively.
* The default index for the vocabulary is set to the index of <unk>.

##### **get_data** method:
* This method converts the tokenized dataset into a format suitable for training a language model.
* For each example in the dataset, the tokens are retrieved, and <eos> is appended to represent the end of the sequence.
* The tokens are then numericalized using the vocabulary **vocab**, and the resulting indices are added to the **data** list.
* The list of indices is converted to a PyTorch LongTensor **torch.LongTensor**.
* The data is reshaped into batches of size **batch_size**, and the function returns the processed data.

**get_data** function is used to preprocess the tokenized datasets for training, validation, and testing.
The resulting train_data, valid_data, and test_data are batches of numericalized sequences ready for input to the language model.



### Model Architecture

**LSTMLanguageModel**, is an LSTM-based language model implemented using PyTorch.
The LSTM model contains following layers:
* **Embedding Layer:** The input of this layer is -- and the output is embedding vector of dimension **emb_dim** which in our case is 1024
* **LSTM Layer:** This layer takes emb_dim(1024) as input and outputs Hidden states with a dimension of hid_dim for each time step in the sequence.

    Parameters for this layer are: 
    * Number of layers: **num_layers**(2)
    * Hidden state dimension: **hid_dim**(1024)
    * Dropout rate: **dropout_rate**(0.65)

    Weights for this layer are initialized uniformly within the range [-init_range_other, init_range_other]
* **Dropout Layer:** This layer is applied to discard some of the output from LSTM layer by randomly setting some of the LSTM output values to 0. Main purpose of this layer is to intoduce regularization. The rate of the outputs to be set as 0 is determined by **droupout_rate**(0.65).
* **Linear (Fully Connected) Layer:** Finally in this layer, output from LSTM layer is fitted which has dimension **hiden_dim**(1024) and this give the score to each word in vocabulary. Here weights are initialized uniformly within the range [-init_range_other, init_range_other] and the bias is set to zero.

The model has a total of 32,695,889 trainable parameters, including the weights and biases in the embedding layer, LSTM layer, linear layer, and other parameters.
       

### Training Process

##### LSTM model's major methods during training

* **Hidden State Initialization:**
The **init_hidden** method is resposible for initialising the hidden state. This method sets hidden state and cell state fro LSTM layer to 0.

* **Forward Method:**
Forward method takes batch of token sequences  and an initial hidden state (hidden) as input. It then embeds the input tokens using the embedding layer and applies dropout to the embedded sequence. Atter that it passes the sequence through the LSTM layer to obtain hidden states.
It then applies dropout to the LSTM output and finally feeds the output through a linear layer to obtain predictions for the next tokens. This method returns the predictions as well as the hidden state.

* **Detaching Hidden State:**
The **detach_hidden** method is used to detach the hidden states from the computation graph for purposes of gradient computation during training.


##### Train Method
* **Loading Data:** Data is loaded to the model in batch with **num_batches**. The batches that are not in multiple of **seq_len** are dropped to maintain sequence length.

* **Training Loop:** In the whole traing iteration following steps are carried out..
    * Training data is iterated in the batch of sequence length
    * Gradient of model parameters is set to 0
    * Hidden state is detached from the computation for better accuracy
    * **get_batch** method is called to get the batch of input and target sequences
    * Data is transferred to specified device (GPU or CPU) for processing
    * Forward pass is made to get the prediction and update the hidden state
    * Loss of prediction is calculated given target
    * Backpropogation is performed
    * **clip_grad_norm** method from torch.nn.utils is used to prevent exploding gradient
    * Model parameters are updated 
    * Loss for the epoch is accumulated

* **Validation**: Similary **evaluate** method is used to get validation loss

* Learning rate is updated using learning rate scheduler
* Finally the model state where the validation loss is best is saved.


Task 3
----------------------------------------------------

### Web app Documentation

##### **Overview**
This is flask web application allows users to input a text prompt and generate a continuation of the text based on a pre-trained LSTM language model.

- This web application consists of two web pages - Home Page(*index.html*) and Result Page(*result.html*)
   * Home Page: 
   ![alt text](./app/static/home-page.png?raw=true)
   Here user can input text prompt to the input box. Additionally user can also determine the length of the text that they want model to generate. User also have the option to select the temperature value from the dropdown and try the result with different values.

   * Result Page: 
   ![alt text](./app/static/result-page.png?raw=true)
   This page show the result that is generated text from the model according to users parameters. It also show the parameter that is input by users and used by model.


##### **Text Generation**
On providing the text prompt, sequence length and temperature in home page to generate the text, **generate** method  of TextGeneration class is called from the **generate.py** file.

- **generate** method is responsible for generating text based on the given prompt using the language model.This method takes input such as:
   * **prompt, max_seq_len, temperature**: *provided by users*
   * **tokenizer**: *torch english okenizer*
   * **model**: *Load from .pt file, which is saved from the best result during training*
   * **vocab**: *loaded from pickle file that is saved during traing on training data*
   * **device**: *Device which is used in model inference (GPU or CPU) based on availability*

##### **Running the Application**
The Flask application is run using python app.py in the terminal, and the web interface can be accessed at http://127.0.0.1:5000/.
