# Text Preprocessing Foundations for LLMs

### Author: Raquel Selma 

This repository contains the implementation and experimentation based on Chapter 2 of the book *"Build a Large Language Model (From Scratch)"* by Sebastian Raschka. The main objective is to understand how Large Language Models (LLMs) process raw text before it enters the neural network.

## Project Content

The main notebook `embeddings.ipynb` covers the following key concepts, including detailed explanations of why each step is crucial for LLMs and agentic systems:

### 1. Data Loading
Reading the raw text (`the-verdict.txt`) and initial preparation. LLMs require text to be converted into a numerical format, so having clean text is the fundamental first step.

### 2. Tokenization
Using the `tiktoken` library (Byte Pair Encoding) to split the text into tokens (subwords). This allows for efficient handling of large vocabularies and unknown words.

<img width="687" height="102" alt="token" src="https://github.com/user-attachments/assets/35166474-cdd7-4e1d-b439-89fa57ec6a86" />

*Figure 1: Output of the tokenization process showing token IDs and their decoding.*

### 3. Data Sampling (Sliding Window)
Creating input-target pairs for model training using a sliding window approach. 

**Experiment:** The `max_length` and `stride` parameters were modified to demonstrate how overlap significantly increases the number of training samples (from 1286 to 2571 samples) and why this helps model generalization.

<img width="1193" height="216" alt="datos " src="https://github.com/user-attachments/assets/be43f32f-6c1c-4f86-aa87-e08d86f106b0" />

*Figure 2: Experiment results comparing different stride values and the number of generated samples.*

### 4. Embeddings (Vector Representation)
Transforming token IDs into continuous vectors (Token Embeddings) and adding positional information (Positional Embeddings). The notebook explains in detail why embeddings encode semantic meaning and their deep relationship with representation learning in neural networks.

<img width="573" height="206" alt="Embeddings" src="https://github.com/user-attachments/assets/89cb7bea-bd8e-450f-8928-f4c36560196c" />

*Figure 3: Shapes of the resulting tensors after applying the embedding layers.*

## Requirements and Execution

1. Clone this repository.
2. Install the necessary dependencies:
   ```bash
   pip install torch tiktoken
   ```
3. Open and run the `embeddings.ipynb` notebook from start to finish.
