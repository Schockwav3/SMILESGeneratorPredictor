# SMILES Generator using DeepSMILES LSTM and Reinforcement Learning with SMILES Predictor

The first part of this project involves a Generator for generating specified SMILES (Simplified Molecular Input Line Entry System) strings using a combined deep learning model. The model leverages deepSMILES, a more compact representation of SMILES and augment it with the SMILES Predictor MLP to enhance the generation process of the LSTM (Long Short Term Model). By combining the base trained LSTM Model with the classification of the SMILES Predictor as Reinforment Learning approach to improve the output of the LSTM and ensure similarity to a target class of molecules.


# SMILES Predictor using a MLP and Morgan Fingerprints, Molecule Descriptors for classification

The second part of this project involves a Predictor. The machine learning model is a Multi-Layer Perceptron (MLP) designed to predict whether a given molecule is an AXL kinase inhibitor based on its SMILES representation. The model leverages various molecular descriptors and Morgan fingerprints to classify the molecules.




## Table of Contents

1. [Introduction](#introduction)
1. [Setup and Installation](#setup-and-installation)
1. [Requirements](#requirements)
2. [Project Structure](#project-structure)
3. [Workflow](#workflow)
4. [SMILES Generator](#smiles-generator)
5. [SMILES Predictor](#smiles-predictor)




## Introduction

The combined approach makes it possible to generate not only a large number of syntactically correct and chemically valid SMILES, but also those that have a high probability of possessing relevant chemical properties. This is done by initially modelling a large number of SMILES data with the SMILES Generator (phase 1) and then fine-tuning them by means of value-driven optimisation using the SMILES Predictor (phase 2). In addition, generated molecules can then be validated separately with the SMILES Predictor for a final quality check (phase 3). This method can be particularly useful in drug design to discover new, potentially effective molecules.




## Setup and Installation

### Conda

   - Please check out the [Conda Documentation](https://github.com/conda/conda-docs).

   - To execute all tasks in one single conda environment the `GeneratorPredictor.yaml` contains all required packages and the corresponding channels
   

     > Note: If you want to update your current environment manually you should add the following **conda packages**:
       >
       > ```bash   
       > - python
       > - cudatoolkit
       > - numpy
       > - pandas
       > - matplotlib
       > - scikit-learn
       > - rdkit
       > - tqdm
       > - pytorch
       > - torchvision
       > - torchmetrics
       > - deepsmiles
       > ```


   - Otherwise navigate to the location of the pulled AMA.yaml file and execute `conda env create -f GeneratorPredictor.yaml`


   - To activate the created conda environmentconda and getting started execute `conda activate GeneratorPredictor`


### Repository

- Clone the repository:

    ```bash
    git clone https://github.com/Schockwav3/SMILESGeneratorPredictor.git
    cd SMILESGeneratorPredictor
    ```


## Requirements

Set up all required data

- **Phase 1 SMILES Generator**

   - You need the input **chembl_smiles.txt** containing the 1.8 million small molecule compounds from the ChEMBL database. This is the dataset what I used to lead the SMILES Generator to train the vocabular and the basic structure of molecules. It is already in the repository. If you want to use a different dataset for training feel free to change it in your project and update the `file_path`. 
   
> [!TIP]
 The defined vocabulary is adjusted to the components of the molecules that occur most frequently in this data set. I have written and added a script `Vocabulary_Creator` that analyses other datasets and outputs which vocabulars are most common. With this information you can extend the vocabulary if necessary. 

   - Once the network has been pretrained on the basic dataset, it will be saved under the `save_model_path` and can then be further trained for Transfer Learning or fine-tuning (Phase 2).


> [!TIP]
 It is also possible to use **Transfer Learning** to a second more specific dataset of SMILES. To do this, simply save the model after the first training and activate `use_pretrained_model = True`, adjust the parameters, define the second / new dataset as new `file_path` and run the code again.


- **Phase 2 SMILES Generator**

   - First you need the pretrained SMILES Generator model from phase 1, simply enter the path under `trained_lstm_path` in the second part of the code.

   - Then you need the pretrained SMILES Predictor model which you will also need in phase 3 for classification. Simply pause here and prepare the learning for the predictor model. Afterwards simply enter the path under `trained_mlp_path` in the second part of the code.

   - Once the network has been finally trained, it will be saved under `save_model_path` in the second part of the code. The fully trained SMILES Generator model can then be used to generate specific SMILES.


- **Phase 3 SMILES Predictor**

   -  You need two files as input. One file **smiles_data** containing different generic molecules enter the path under `smiles_data_path` and one file with targets **smiles_axl_inhibitors** that the model should learn to predict. In this case, we want to train the model to distinguish whether a molecule is an AXL kinase inhibitor or not. Update the path under `smiles_axl_inhibitors_path`.

    Target = 1: AXL kinase inhibitor
    Target = 0: Other molecules

> [!TIP] 
You should search for as much target SMILES for your prediction class input as possible. In my example I found a total of 4564 on ChEMBL and NIH. For the generic smiles data I took a ratio of around 1:2 so a total of 10812 random non target SMILES. 


   - Once the network has been trained, it will be saved under `save_model_path`. The trained SMILES Predictor model can then be used to classify SMILES.




## Project Structure

This will give you a complete overview of the **SMILES GeneratorPredictor** project structure, all existing scripts in the repository and all required files:

![Project Structure](https://github.com/Schockwav3/SMILESGeneratorPredictor/blob/main/Pictures/project_structure.png)




## Workflow

This will give you a complete overview of the **SMILES GeneratorPredictor** Workflow:

<img src="https://github.com/Schockwav3/SMILESGeneratorPredictor/blob/main/Pictures/workflow.png" width="600" height="1360">




## SMILES Generator using DeepSMILES LSTM and Reinforcement Learning with SMILES Predictor


### Device Selection

The primary purpose of device selection is to determine whether the computations will be performed on a CPU or a GPU. The choice of device can significantly impact the training and inference speed of deep learning models, especially those involving large datasets and complex architectures.

- **Checking GPU Availability:** The code starts by checking if a CUDA-capable GPU is available on the machine. This is done using the function `torch.cuda.is_available()`.

- **If a GPU is Available:** If a CUDA-compatible GPU is found, the device is set to cuda using `torch.device("cuda")`. This setting indicates that the model and its computations will be transferred to the GPU, which can provide significant speed-ups due to its parallel processing capabilities.

- **If a GPU is Not Available:** If no compatible GPU is detected, the device is set to cpu using `torch.device("cpu")`. This means that all computations will be performed on the CPU.




### Define the Vocabulary

The `class FixedVocabulary` defines a fixed set of tokens for encoding SMILES sequences.

```bash  
           'PAD':      0,       # Padding token (for filling batches of unequal length)
           'UNK':      1,       # Undefined token (for unknown elements)
           '^':        2,       # Start token
           '$':        3,       # End token
           '3':        4,
           '4':        5,
           '5':        6,
           '6':        7,
           '7':        8,
           '8':        9,
           '9':        10,
           '%10':      11,
           '%11':      12,
           '%12':      13,
           '%13':      14,
           '%14':      15,
           '%15':      16,
           '%16':      17,
           '%17':      18,
           '%18':      19,
           '%19':      20,
           '%20':      21,
           '%21':      22,
           '%22':      23,
           '%23':      24,
           '%24':      25,
           ')':        26,
           '=':        27,
           '#':        28,
           '.':        29,
           '-':        30,
           '/':        31,
           '\\':       32, 
           'n':        33,
           'o':        34,
           'c':        35,
           's':        36,
           'N':        37,
           'O':        38,
           'C':        39,
           'S':        40,
           'F':        41,
           'P':        42,
           'I':        43,
           'B':        44,
           'Br':       45,
           '[C@]':     47,
           '[C@H]':    48,
           '[C@@]':    50,
           '[nH]':     51,
           '[O-]':     52,
           '[N+]':     53,
           '[n+]':     54,
           '[Na+]':    55,
           '[S+]':     56,
           '[Br-]':    57,
           '[I-]':     59,
           '[N-]':     60,
           '[Si]':     61,
           '[2H]':     62,
           '[K+]':     63,
           '[Se]':     64,
           '[P+]':     65,
           '[C-]':     66,
           '[se]':     67,
           '[Cl+3]:':  68,
           '[Li+]:':   69,      
```

> [!TIP]
If you need to update or change the `class FixedVocabulary` you can use the sript in /src/Vocabulary_Creator.ipynb to analyze a file with SMILES and see which Tokens are used and how many of them are included to create a updated Vocabulary but for most use cases this Vocabulary should be fine.




### Tokanizer

- The `class DeepSMILESTokenizer` handles the transformation of SMILES into deepSMILES and performs tokenization and untokenization.

> [!TIP]
**DeepSMILES** is an extension of the standard SMILES format (Simplified Molecular Input Line Entry System) for the visualisation of molecular structures. While SMILES describes the structure of a molecule using a character string, DeepSMILES goes one step further by simplifying and compressing this representation. They make it possible to increase the learning curve in machine learning by reducing the complexity and length of the representations without losing information about the chemical structure.

- The Tokanizer uses several regular expressions to tokenize deepSMILES strings. Each regex pattern is designed to  match specific components of a SMILES string. Below are the regex patterns used and their purposes:

    - `brackets` groups characters within square brackets together, which can represent charged atoms or specific configurations in the SMILES syntax.

    - `2_ring_nums` matches numbers up to two digits preceded by a percent sign ("%"), used to denote ring closures in molecules with more than 9 rings.

    - `brcl` matches the halogen atoms bromine ("Br") and chlorine ("Cl"), ensuring they are recognized as unique tokens in the SMILES string. They are essential in drug molecules.


 ```bash
       "brackets": re.compile(r"(\[[^\]]*\])"),
       "2_ring_nums": re.compile(r"(%\d{2})"),
       "brcl": re.compile(r"(Br|Cl)")
 ```




### Define the LSTM Model (RNN)

- The LSTM base model is designed to handle the generation and manipulation of SMILES representations using an **RNN (Recurrent Neural Network)** architecture with **LSTM (Long Short-Term Memory)** cells.

- **LSTM model structure**:

    - **embedding layer** which converts the input sequences into dense vectors.

    - **LSTM layers** which processes these vectors sequentially and returns a sequence of outputs.

    - **linear layer** at the end transforms the LSTM outputs back to the size of the vocabulary so that the probabilities for the next token can be calculated.

- **Training process:** 

    - During training, the model receives a SMILES sequence and learns to predict the next token in the sequence. The training loss is calculated by measuring the difference between the predicted and actual tokens.

- **Sequence generation:**

    - During generation, the model begins with a start token (^) and progressively predicted the next token until an end token ($) is reached or a maximum sequence length is reached.

    - `generate_deepsmiles(num_samples, max_length)`: Generates a specified number of deepSMILES sequences up to a maximum length, used for creating new molecular representations.

    - `convert_deepsmiles_to_smiles(deep_smiles_list)`: Converts a list of deepSMILES sequences back to SMILES format, making the output interpretable in a chemical context.

- The input parameters allow users to configure the model according to the complexity of the dataset and the computational resources available. The model's capability to load pretrained weights also facilitates fine-tuning (phase 2) and Transfer Learning, making it adaptable to new tasks with minimal retraining.


 ```bash
       default_params = {
        "layer_size": 512,
        "num_layers": 3,
        "embedding_layer_size": 128,
        "dropout": 0.0,
        "layer_normalization": False
        }
 ```

- Key Input Parameters:

   - `layer_size (int)`: The number of units in each LSTM layer, determining the model's capacity.

   - `num_layers (int)`: The number of LSTM layers stacked in the model, affecting the depth and representational power.

   - `embedding_layer_size (int)`: The size of the embedding vectors that represent input tokens, influencing the richness of token representations.

   - `dropout (float)`: The dropout rate applied to prevent overfitting by randomly setting some LSTM outputs to zero during training.

   - `layer_normalization (bool)`: A flag indicating whether to apply layer normalization, which helps stabilize and accelerate training.




### Define the Trainer

- The SmilesTrainer class is used to streamline the training process of the LSTM model. It handles data loading, model training, validation, and testing, as well as monitoring the training progress through loss and accuracy metrics. The class can utilize a pretrained model for fine-tuning (phase 2), making it versatile for different stages of model development. The plotting functions provide a clear visualization of the training dynamics, allowing for easy identification of issues such as overfitting or underfitting.

- **NLLLoss:**
    - `nn.NLLLoss()` stands for the **"Negative Log Likelihood Loss "**. It is a loss function that is often used in classification problems. It calculates the negative log-likelihood of the correct class. If the probability of the correct class is low, the log value becomes negative and large, resulting in a high loss. The loss is minimised by training the model to maximise the probability of correct classes.

- **Calculate predictions:**
    - `outputs.argmax(dim=-1)` selects the class with the highest log probability for each position in the sequence. The result is a tensor where each position contains the predicted class.

- **Determine correct predictions:**
    - `(predictions == targets).float()` compares the predictions with the actual target classes and produces a tensor containing 1 if the prediction is correct and 0 if it is incorrect.

- **Calculate accuracy:**
    - `correct.mean()` calculates the average of the correct predictions, which corresponds to the accuracy, i.e. the proportion of correctly predicted tokens.

- These methods are crucial for monitoring and optimising the training process as they provide insights into the performance of the model. Loss helps to determine the direction for updating the model parameters, while accuracy is a direct metric for the model's performance in terms of correct predictions.

- The division into the three data sets is important to ensure a fair and reliable evaluation of model performance. Without this, the model could be over-optimised on the training data and subsequently perform poorly on new data.

    - `train` is used to train the model. The model learns patterns, relationships and structures in the data and adjusts its parameters to improve predictions.

    - `validation` is used to evaluate the model during training. They help to assess the performance of the model independently of the training data.

    - `test` is only used to evaluate the final model performance after the training has been completed.

- By plotting the loss and accuracy metrics over the training epochs, you can gain deeper insights into the behaviour of the model during the training process, evaluating model performance and react accordingly. 


```bash
def plot_losses(self):
    plt.figure(figsize=(10, 5))
    plt.plot(self.train_losses, label='Training loss')
    plt.plot(self.valid_losses, label='Validation loss')
    plt.plot(self.test_losses, label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
```

```bash
def plot_accuracy(self):
    plt.figure(figsize=(10, 5))
    plt.plot(self.train_accuracy, label='Training accuracy')
    plt.plot(self.valid_accuracy, label='Validation accuracy')
    plt.plot(self.test_accuracy, label='Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
```

- `plot_losses`: Visualises the progress in loss during training.

- `plot_label_accuracy`: Visualises the accuracy for different labels over the training time.



```bash
def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, 
                 epochs=20, learning_rate=0.0010, batch_size=250, use_pretrained_model=None, load_model_path=None, save_model_path=None):
```

- Key Input Parameters:

    - `epochs (int)`: The number of epochs (full passes through the training dataset) to train the model.

    - `learning_rate (float)`: The learning rate for the optimizer, controlling how much to adjust the model's weights with respect to the loss gradient.

    - `batch_size (int)`: The number of samples per batch to be loaded by the DataLoader.




### Define the Dataset

- The SMILESDataset class is designed to preprocess SMILES data for the machine learning task. It efficiently handles data augmentation, tokenization, and encoding to prepare input-target pairs for model training. 

- **Data Augmentation** enhances the diversity of the dataset, potentially improving model robustness. For each SMILES in the list, if `augment=True`, it generates multiple random valid SMILES representations of the same molecule using `randomize_smiles(self, smile: str, num_random: int)` method. The model can learn to recognize the molecule regardless of the specific SMILES string used.

- `load_data`: reads a file containing SMILES strings (one per line) and returns a list of these strings.

- `split_data`: splits a given dataset into three subsets: **training=0.7**, **validation=0.15**, and **test=0.15** sets.


```bash
def __init__(self, smiles_list, tokenizer, vocabulary, augment: bool = True, augment_factor: int = 4):
```

- Key Input Parameters:

    - `augment (bool)`: A flag indicating whether to apply data augmentation to the SMILES strings.

    - `augment_factor (int)`: The number of augmented versions to generate for each SMILES string.




### File Path


```bash
file_path = '/****/****/chembl_smiles.txt'
```

- `file_path (str)`: Path to the SMILES file which includes the dataset for prelearning the SMILES Generator model. 

- If Transfer Learning is activated, update the path to the new SMILES file which the pre-trained model should retrain. 




### Save, Load and Transfer Learning

- These settings determine whether an already trained model should be used and where this model should be saved or loaded. This is useful for the application of Transfer Learning, where a model is based on previously learnt weights to reduce training time and improve performance. Or atleast to save the pre-trained model.


```bash
use_pretrained_model = True
load_model_path = '/****/****/model_new_try0907.pth'
save_model_path = '/****/****/model_new_try0408.pth'
```

- `use_pretrained_model (bool)`: If this variable is set to `True`, the model is not trained from scratch, it indicates that a pre-trained model should be used. It loads and uses weights that have already been learnt. Can be used for Transfer Learning or the fine-tuning (phase 2) process.

- `load_model_path (str)`: Specifies the file path where the pre-trained saved model is located.

- `save_model_path (str)`: This path specifies where the model is to be saved after pre-training or Transfer Learning. It should always be filled with a valid path otherwise the model won`t be saved.




### Activate_fine_tuning (Phase 2)

- Control variable for activating fine-tuning (Part 2)

- `activate_fine_tuning = False` if you want to pretrain the SMILES Generator model (Part 1) or Transfer Learning and skip the second part of the code
- `activate_fine_tuning = True` if you want to enter fine-tuning (Part 2)


## Phase 2


### Define pre-trained Models

- Provide the paths to the saved pre-trained LSTM model which will be now be used for further training (fine-tuning) and the already trained SMILES predictor that will be used for the evaluation of the reinforcement learning rewards.

- `trained_lstm_path (str)`: File path to the pre-trained LSTM model

- `trained_mlp_path (str)`: Path to the trained MLP model. The MLP model is used here as a predictor to evaluate the quality of the sequences generated by the LSTM model

> [!TIP]
If the SMILES Predictor isn't ready pause here and continue later after training the MLP watch workflow or follow the next section **SMILES Predictor using a MLP and Morgan Fingerprints, Molecule Descriptors for validation**




### MLP Definition & Descriptors

- `class MLP(nn.Module)`: Loads and defines the multilayer perceptron (MLP) from the SMILES predictor. The output is a two-class classification.

- `def calculate_descriptors(smiles)`: Converts a SMILES string into an RDKit molecule object and calculates a variety of chemical descriptors.

- `def evaluate_smiles(smiles, mlp_model, tokenizer, vocabulary)`: Evaluates them with the MLP and returns the prediction. The function returns either **1 (AXL-classified)** or **0 (not AXL-classified)**, depending on the result of the MLP prediction.




### Define The Trainer Phase2

- This class implements the fine-tuning of the SMILES Generator model with a reinforcement learning approach based on classification by the SMILES Predictor model.

1. **Sampling:** The model generates a series of sequences (DeepSMILES).
    - `generated_deepsmiles = self.model.generate_deepsmiles(num_samples=1, max_length=100)[0]` Generates DeepSMILES sequences and converts them to SMILES `generated_smiles = self.model.convert_deepsmiles_to_smiles([generated_deepsmiles])[0]`. Validates the generated SMILES and sorts out invalid ones.

2. **Rewarding:** Each sequence receives a reward based on its quality.

    - Evaluates the SMILES generated using the MLP model.

    - Apply higher rewards `reward = 2.5` for AXL-classified SMILES `score == 1`.

    - Give a small reward `reward = 0.5` for valid but not AXL-classified SMILES `score == 0`.

    - Uses penalties `reward = -7.5` for invalid SMILES.


```bash
if score == 1:
    axl_classified_count += 1
    axl_classified_smiles_list.append(generated_smiles)
    reward = 2.5
```

```bash
else:
    non_axl_classified_count += 1
    non_axl_classified_smiles_list.append(generated_smiles)
    reward = 0.5 
```

```bash
else:
    invalid_smiles_count += 1
    invalid_smiles_list.append(generated_smiles)
    reward = -7.5
```

3. **Log-Probs calculation:** For each sequence, the log probability of its generation is calculated.

    - `logit, _ = self.model(sequence)`: The generated sequence input `(sequence)` is passed through the model `(self.model)`. The model returns `logits`, which represent the raw values of the probabilities over the entire vocabulary for each position in the sequence. These `logits` are passed through the log softmax function `(F.log_softmax)` to calculate the log probabilities `(log_probs)`. For each position in the sequence, the log probability of the actual generated token is extracted. 
    
    - `log_probs.append(log_prob.sum())`: The sum of the log probabilities for the entire sequence is calculated and added to `log_probs`. This sum represents the log probability of the entire generated sequence.

4. **Loss calculation:** A policy loss is calculated, which indicates how the model should adjust its probabilities in order to obtain more positive rewards in the future.

    - `policy_loss = -log_probs * rewards`: The product of log_probs and rewards gives the value that indicates how well the model generated this sequence based on the reward received. The negative sign (`-`) is used because we want to minimise the loss, but the goal is to maximise the rewards (higher rewards for better sequences).

    - `policy_loss = policy_loss.mean()`: The average policy loss over all generated sequences is calculated. This average value is used as the final loss that the model wants to minimise.

5. **Backpropagation:** The calculated loss is used for the backpropagation to calculate the gradients and update the weights of the model.

    - `self.optimizer.zero_grad()`: Before calculating the gradients, all previously calculated gradients are set to zero

    - `policy_loss.backward()`: Calculates the gradients of the policy loss with respect to all parameters in the model. These gradients indicate the direction and amount of adjustment required for each model parameter to minimise the loss.

6. **Optimisation:** The optimiser parameters are updated to improve the model.

    - `self.optimizer.step()`: After the gradients have been calculated, the optimiser `Adam` adjusts the model parameters based on these gradients.


```bash
trainer_phase2 = SmilesTrainerPhase2(
        epochs=120,
        learning_rate=0.00002,
        batch_size=32,
        save_model_path='/****/****/finetuned_model2507.pth',
        num_generated_smiles=400,
        reward_scale=1
    )
```

- Key Input Parameters:

    - `num_generated_smiles`: Number of SMILES generated per epoch.

    - `epochs`: Number of training runs.

    - `learning_rate`: Learning rate for the optimiser.

    - `batch_size`: Batch size for the training.

    - `save_model_path`: Path to save the fine-tuned model.

    - `reward_scale`: Scalar for adjusting the rewards during fine-tuning.


- `calculate_accuracy`: Calculates the accuracy as the ratio of AXL-classified SMILES to the total number of valid SMILES generated.

- `calculate_total_accuracy`: Calculates the total accuracy as the ratio of AXL-classified SMILES to all generated SMILES.

- Logs the number of generated, valid and AXL-classified SMILES as well as the total rewards.



### Plotting

- By plotting the generated and classified count and accuracy metrics over the training epochs, you can gain deeper insights into the behaviour of the model during the training process, evaluating model performance and react accordingly. 


```bash
    def plot_accuracy(train_accuracy, total_accuracy, epochs):
        plt.figure(figsize=(14, 6))
        plt.plot(epochs, train_accuracy, label='Training Accuracy')
        plt.plot(epochs, total_accuracy, label='Total Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Total Accuracy Progress')
        plt.legend()
        plt.grid(True)
        plt.show()
```

```bash
    def plot_generated_smiles(generated_smiles_counts, axl_classified_counts, epochs):
        plt.figure(figsize=(14, 6))
        plt.plot(epochs, generated_smiles_counts, label='Generated SMILES')
        plt.plot(epochs, axl_classified_counts, label='AXL Classified SMILES')
        plt.xlabel('Epochs')
        plt.ylabel('Count')
        plt.title('Generated SMILES and AXL Classified SMILES Count')
        plt.legend()
        plt.grid(True)
        plt.show()

    plt.tight_layout()
    plt.show()
```

- `plot_accuracy`: Visualises the accuracy of the model during training. The training accuracy and the overall accuracy are visualised.

- `plot_generated_smiles`: Visualises the ratio of generated SMILES and AXL-classified SMILES across the epochs.


## SMILES Predictor using a MLP and Morgan Fingerprints, Molecule Descriptors for validation


### Device Selection

The primary purpose of device selection is to determine whether the computations will be performed on a CPU or a GPU. The choice of device can significantly impact the training and inference speed of deep learning models, especially those involving large datasets and complex architectures.

- **Checking GPU Availability:** The code starts by checking if a CUDA-capable GPU is available on the machine. This is done using the function `torch.cuda.is_available()`.

- **If a GPU is Available:** If a CUDA-compatible GPU is found, the device is set to cuda using `torch.device("cuda")`. This setting indicates that the model and its computations will be transferred to the GPU, which can provide significant speed-ups due to its parallel processing capabilities.

- **If a GPU is Not Available:** If no compatible GPU is detected, the device is set to cpu using `torch.device("cpu")`. This means that all computations will be performed on the CPU.




### Load Input Data

- `def load_data(file_path)`: Reads in a text file containing a list of SMILES strings.

- `smiles_data_path (str):` Path for the input file, which contains a general collection of SMILES strings.

- `smiles_axl_inhibitors_path (str):` Path for a second input file containing only AXL kinase inhibitors SMILES.




### Descriptors

- `calculate_descriptors`: Function that calculates chemical descriptors for a given molecule. Descriptors are numerical values that represent certain physical, chemical or structural properties of a molecule. These properties can be used in machine learning models to make predictions about the activity, toxicity or other characteristics of molecules.

- `Chem.MolFromSmiles(smiles)`: Function converts a SMILES string into an RDKit molecule object (`mol`). If the SMILES string is invalid and no molecule object can be created, `None` is returned. This means that the SMILES string was invalid and no descriptors can be calculated.

**The following descriptors are calculated for each molecule:**

- **Morgan Fingerprints**: A type of molecular fingerprint used for similarity searching, based on the circular substructures around each atom, and encoded as binary vectors. Morgan fingerprints are particularly useful in capturing the local environment of atoms within a molecule. They are generated by considering circular neighborhoods of varying radii around each atom, and then hashing these neighborhoods into fixed-length binary vectors.
- **AlogP**: A measure of the lipophilicity (fat-loving nature) of a molecule, representing its ability to dissolve in fats, oils, lipids, and non-polar solvents.
- **Polar Surface Area**: The total area of a molecule that is polar (i.e., capable of hydrogen bonding), influencing drug absorption and transport properties.
- **HBA (Hydrogen Bond Acceptors)**: The number of hydrogen bond acceptor sites within a molecule, affecting its solubility and interaction with biological targets.
- **HBD (Hydrogen Bond Donors)**: The number of hydrogen bond donor sites within a molecule, influencing solubility and biological interactions.
- **Bioactivities (pki Value)**: A measure of the potency of a molecule in inhibiting a specific biological target, expressed as the negative logarithm of the inhibition constant (Ki).
- **Chi0**: A molecular connectivity index representing the molecule's overall structure.
- **Kappa1**: A shape index representing the molecule's flexibility.
- **TPSA (Topological Polar Surface Area)**: The surface area of a molecule contributed by polar atoms, influencing drug absorption and permeability.
- **MolLogP**: The logarithm of the partition coefficient between water and octanol, representing a molecule's hydrophobicity.
- **PEOE_VSA1 to PEOE_VSA14**: A series of descriptors representing the van der Waals surface area of a molecule, divided into regions of different partial charges, which influence molecular interactions.
- **Molecular Weight**: The total mass of a molecule, influencing its distribution and elimination from the body.
- **NumRotatableBonds**: The number of bonds in a molecule that can rotate, affecting its flexibility and interaction with biological targets.
- **NumAromaticRings**: The number of aromatic (ring-shaped) structures within a molecule, influencing its stability and interaction with biological targets.
- **FractionCSP3**: The fraction of sp3 hybridized carbons in a molecule, representing the degree of saturation and three-dimensionality.
- **Polarizability**: A measure of how easily the electron cloud around a molecule can be distorted, influencing its interaction with electric fields and other molecules.
- **MolVolume**: The volume occupied by a molecule, affecting its density and interaction with biological environments.
- **MolWt**: Another term for molecular weight, indicating the total mass of a molecule.
- **HeavyAtomCount**: The number of non-hydrogen atoms in a molecule, influencing its size and reactivity.
- **NHOHCount**: The number of nitrogen, hydrogen, and oxygen atoms in a molecule, which can affect its polarity and reactivity.
- **NOCount**: The number of nitrogen and oxygen atoms in a molecule, influencing its hydrogen bonding and reactivity.
- **NumHeteroatoms**: The number of atoms in a molecule that are not carbon or hydrogen, affecting its reactivity and interaction with biological targets.
- **NumRadicalElectrons**: The number of unpaired electrons in a molecule, which can influence its reactivity and stability.
- **NumValenceElectrons**: The number of electrons in the outer shell of a molecule's atoms, determining its reactivity and bonding behavior.
- **RingCount**: The number of ring structures within a molecule, affecting its stability and interaction with biological targets.
- **BalabanJ**: A topological index representing the overall shape and branching of a molecule.
- **BertzCT**: A complexity index representing the structural complexity of a molecule.
- **Chi1**: A molecular connectivity index representing the molecule's overall structure, similar to Chi0 but focusing on different aspects.
- **Chi0n**: A molecular connectivity index representing the molecule's overall structure, focusing on nitrogen atoms.
- **Chi0v**: A molecular connectivity index representing the molecule's overall structure, focusing on valence electrons.
- **Chi1n**: A molecular connectivity index representing the molecule's overall structure, focusing on the first level of nitrogen atoms.
- **Chi1v**: A molecular connectivity index representing the molecule's overall structure, focusing on the first level of valence electrons.
- **Kappa2**: A shape index representing the molecule's flexibility and three-dimensionality.
- **Kappa3**: A shape index representing the molecule's overall three-dimensional structure.
- **HallKierAlpha**: An index representing the overall branching and complexity of a molecule's structure.


- Calculation of molecular descriptors for two different sets of SMILES data: general molecules and specific AXL kinase inhibitors. The calculated descriptors are then stored in a Pandas DataFrame, together with Morgan fingerprints extracted into separate columns.

1. `data = []`: An empty list data is initialised to store the descriptors for all molecules.

    - The **target** variable indicates whether a molecule is an AXL kinase inhibitor or not:


```bash
Target = 1: AXL kinase inhibitor
Target = 0: Other molecules
```

2. Calculation of the descriptors for general SMILES data:

    - `for index, smiles in enumerate(smiles_data)`: Iterates through the list smiles_data, where index is the current index and smiles is the current SMILES string.

    - `descriptors = calculate_descriptors(smiles)`: Calculates the descriptors for the current SMILES string using the calculate_descriptors function.

    - `if descriptors is not None`: Checks whether the calculation was successful (i.e. the SMILES string was valid).

    - `descriptors["SMILES"] = smiles`: Adds the SMILES string to the descriptors.

    - `descriptors["Target"] = 0`: Sets the **target label to 0** to indicate that it is a generic molecule.

    - `data.append(descriptors)`: Adds the descriptor dictionary to the data list.


```bash
#example
- smiles.txt:
    
    CCO
    CCC
    COC
```

```bash
#example
- smiles.txt`:

    CCO: Descriptors + Target = 0
    CCC: Descriptors + Target = 0
    COC: Descriptors + Target = 0
```

3. Calculation of the descriptors for AXL kinase inhibitors:

    - `for index, smiles in enumerate(smiles_axl_inhibitors)`: Iterates through the list smiles_axl_inhibitors.

    - `descriptors = calculate_descriptors(smiles)`: Calculates the descriptors for the current SMILES string.
    
    - `if descriptors is not None`: Checks whether the calculation was successful.

    - `descriptors["SMILES"] = smiles`: Adds the SMILES string to the descriptors.

    - `descriptors["Target"] = 1`: Sets the **target label to 1** to indicate that it is an AXL kinase inhibitor.

    - `data.append(descriptors)`:Adds the descriptor dictionary to the data list.


```bash
#example
- axl_inhibitors.txt:
    
    C1=CC=CC=C1
    C2=CCN=C(C)C2
```

```bash
#example
- axl_inhibitors.txt:

    C1=CC=CC=C1: Descriptors + Target = 1
    C2=CCN=C(C)C2: Descriptors + Target = 1
```

4. Creation of a DataFrame:

    - `df = pd.DataFrame(data)`: Converts the list data into a Pandas DataFrame df, where each row corresponds to a molecule and each column to a descriptor.

5. Processing of the Morgan fingerprints:

    - `fingerprints = np.array([list(fp) for fp in df["MorganFingerprint"].values])`: Extracts the Morgan fingerprints from the DataFrame `df` and converts them into a numpy array fingerprints. Each fingerprint is converted into a list of bits.

6. Merge the main DataFrame with the fingerprint data:

    - `df = df.drop(columns=["MorganFingerprint"])`: Removes the original MorganFingerprint column from `df` as the fingerprint data is moved to `fingerprints_df`.

    - `df = pd.concat([df, fingerprints_df], axis=1)`: Adds `fingerprints_df` to the main DataFrame `df` by appending the fingerprint columns to the main DataFrame.

- The resulting DataFrame df contains all calculated molecular descriptors, including the extracted Morgan fingerprints, for each molecule in the two SMILES datasets.




### Define the Dataset

- Process of splitting the data into training and test sets, converting this data into PyTorch tensors and creating DataLoaders to efficiently process the data during training

- `X = df.drop(columns=["SMILES", "Target"])`: Removes the columns "SMILES" and "Target" from the DataFrame `df` to create the feature matrix `X`. This matrix contains all numerical descriptors and fingerprints that serve as input data for the model.

- `y = df["Target"]`: Extracts the target variable `y` from the DataFrame, which indicates whether the molecules are general molecules (`0`) or AXL kinase inhibitors (`1`).

- `train_test_split(X, y, test_size=0.2, random_state=42)`: Splits the data into training and test sets. `test_size=0.2` means that 20% of the data is used as test data. `random_state=42` ensures that the division is reproducible.

- Converts the Pandas DataFrames `X_train`, `y_train`, `X_test`, and `y_test` into PyTorch tensors.

- DataLoader for train and test data will be created


```bash
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

- Key Input Parameters:

- `Batch_size (int)`: Sets the size of the batches which means that the data is processed in batches of x each.




### Define the MLP model


```bash
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 8)
        self.bn5 = nn.BatchNorm1d(8)
        self.fc6 = nn.Linear(8, 2)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x
```

- A **Multi-Layer Perceptron (MLP)** is used to process the descriptors and learn to classify molecules as AXL kinase inhibitors or not. An MLP is a feedforward artificial neural network that is used to classify molecules into two categories. It consists of several **fully connected layers**, **batch normalisation** and **dropout** to improve model performance and avoid overfitting.

1. Input layer: The input layer receives vectors of molecular descriptors, the number of which is determined by the number of features in the training data.

2. Hidden layers: There are a total of five fully connected dense hidden layers in the architecture:

    - **Activation function**: a ReLU (Rectified Linear Unit) is applied to perform non-linear transformations.

    - **Batch normalisation**: A batch normalisation layer is inserted after each fully connected layer. These layers normalise the outputs of the previous layers, which can improve the stability and speed of the training process.

    - **Dropout**: This is a regularisation technique in which a certain rate of neurons is randomly deactivated during training. This helps to prevent overfitting.

3.  Output layer: The final output layer of the MLP consists of 2 neurons. These represent the two classes of the classification problem (e.g. AXL kinase inhibitor and non-inhibitor). This layer outputs raw scores (`logits`), which can then be used to calculate probabilities for the classes.

> [!TIP]
You can adjust the layer-size or add more layers to the MLP if you want to improve the results on your own dataset.




### Class Weights

- `class_weights`: A tensor that contains the weights for the classes in the training data. These are used to equalise the imbalance between the classes. The positive examples are given a higher weight to compensate for their relatively lower occurrence compared to negative examples.

- `focal_CE_loss`: This function calculates the focal loss, a modified version of the cross-entropy loss function. It focusses the training on examples that are difficult to classify.

    - `scores`: The predictions of the model (raw scores or logits before applying the softmax function).

    - `labels`: The actual labels of the data.

    - `gamma`: A hyperparameter that controls the influence of the examples that are difficult to classify. Higher values of gamma increase the focus on these examples.


```bash
criterion = F.cross_entropy
criterion = focal_CE_loss
optimizer = optim.Adam(model.parameters(), lr=0.00005)
```

- `optimizer = optim.Adam(model.parameters())`: Defines the optimiser that updates the weights of the model based on the calculated gradients. The **Adam optimiser** is used here

- Key Input Parameters:

    - `lr`: Sets the learning rate to a small value to ensure stable and slow weight updates, which is particularly important to avoid overfitting to the training data here.




### Define the Trainer

- The trainer carries out the complete training of the model, monitors the performance and saves the results for later analysis and visualisation.


```bash
num_epochs = 50
```

- Key Input Parameters:

    - `num_epochs`: Number of training runs.


- **Epoch Initialisation:**
    
    - At the beginning of each epoch, the model is set to training mode `model.train()`.

    - The total loss `running_loss` is initialised. It accumulates the loss for the entire epoch so that the average loss can be calculated later.

- **Training Loop:**

    - **Reset gradient:** Before a new step of the backpropagation process begins, the gradient must be set to zero `optimiser.zero_grad()`.

    - **Prediction and loss calculation:** The model processes the input data (`inputs`) and generates outputs (`outputs`). These outputs are compared with the target values (`labels`) to calculate the loss (`loss`)
    
    - **Backward propagation and weight update:** loss.backward() calculates the gradients of the loss function. The optimiser (`optimizer.step()`) updates the weights of the model based on these gradients to minimise the loss.
    
    - **Add up total loss:** The loss for the current batch is added to `running_loss`. Multiplying by `inputs.size(0)` ensures that the loss is correctly weighted for the number of samples in the batch.

- **Loss Calculation:**

    - After processing all batches in an epoch, the average loss is calculated `epoch_loss = running_loss / len(train_loader.dataset)`. This average loss (`epoch_loss`) indicates how well the model can predict the training data.

- **Training Accuracy Calculation:**

    - After completing the training process for an epoch, the model is set to evaluation mode (`model.eval()`). The predictions of the model (`outputs`) are compared with the actual labels (`labels`) to count the number of correct predictions (`train_correct`) and the total number of examples (`train_total`). The accuracy for each class is then calculated `train_accuracy = {label: 100 * train_correct[label] / train_total[label]}`, which gives the percentage of correct predictions in relation to the total number of examples of that class.

- **Test Accuracy Calculation:**

    - After calculating the training accuracy, a similar procedure is performed on the test data set. The test accuracy provides information on how well the model performs on unseen data and is also logged separately for each class. This evaluation is crucial for recognising overfitting




### Save Model


```bash
save_model_path = 'model_predictor_0407.pth'
```

- `save_model_path (str)`: This path specifies where the model is to be saved after training. It should always be filled with a valid path otherwise the model won`t be saved.




### Plotting

- By plotting the generated and classified count and accuracy metrics over the training epochs, you can gain deeper insights into the behaviour of the model during the training process, evaluating model performance and react accordingly. 


```bash
def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
```

```bash
def plot_label_accuracy(label_accuracies, label_name):
    plt.figure(figsize=(10, 5))
    for label in [0, 1]:
        plt.plot(label_accuracies[label], label=f'{label_name} Accuracy - Label {label}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{label_name} Accuracy Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
```

```bash
def plot_confusion_matrix(mdl, data, 
                          class_names=None, device=torch.device("cpu")):

    preds = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    mdl = mdl.to(device)
    mdl.eval()

    pbar = tqdm(data, desc="predict", file=sys.stdout)
    for x,y in pbar:
        x,y = x.to(device), y.to(device)
        p = mdl(x)
        preds = torch.cat((preds, torch.argmax(p, dim=1)), dim=0)
        targets = torch.cat((targets, y), dim=0)

    preds, targets = preds.to("cpu"), targets.to("cpu")
    print(preds.shape, targets.shape)
    cm = MulticlassConfusionMatrix(num_classes=2)
    cm.update(preds, targets)
    fig, ax = cm.plot(labels=class_names)
    return fig
```

- `plot_losses`: Visualises the progress in loss during training.

- `plot_label_accuracy`: Visualises the accuracy for different labels over the training time.

- `plot_confusion_matrix`: Visualises a confuion matrix for the predictions of a model compared to the actual labels. A confuion matrix is an important tool for evaluating the classification performance of a model as it shows the number of correctly and incorrectly classified examples for each class.




--------------------------------------------------------



## References

- RDKit: Open-source cheminformatics software.
- DeepSMILES: An alternative compact representation of SMILES.
- PyTorch: An open-source machine learning library.


## Licence
