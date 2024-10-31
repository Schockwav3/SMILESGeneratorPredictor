# DeepSMILESGenerator using DeepSMILES LSTM and Reinforcement Learning with DeepSMILESPredictor

The first part of this project involves a DeepSMILESGenerator for generating specified SMILES (Simplified Molecular Input Line Entry System) strings using a combined deep learning model. The model leverages deepSMILES, a more compact representation of SMILES and augment it with the DeepSMILESPredictor MLP to enhance the generation process of the LSTM (Long Short Term Model). By combining the base trained LSTM Model with the classification of the DeepSMILESPredictor as Reinforment Learning approach to improve the output of the LSTM and ensure similarity to a target class of molecules.


# DeepSMILESPredictor using a MLP and ECFPs, Molecule Descriptors for classification

The second part of this project involves a DeepSMILESPredictor. The machine learning model is a Multi-Layer Perceptron (MLP) designed to predict whether a given molecule is an AXL kinase inhibitor based on its representation. The model leverages various molecular descriptors and Extended-Connectivity Morgan fingerprints to classify the molecules.




## Table of Contents

1. [Introduction](#introduction)
1. [Setup and Installation](#setup-and-installation)
1. [Requirements](#requirements)
2. [Training Structure](#training-structure)
3. [Project Workflow](#project-workflow)
4. [Train DeepSMILESPredictor](#train-deepsmilespredictor-using-a-mlp-and-ecfps-molecule-descriptors-for-validation)
5. [Train DeepSMILESGenerator](#train-deepsmilesgenerator-using-deepsmiles-lstm-and-reinforcement-learning-with-deepsmilespredictor)
6. [Master Thesis](#master-thesis)
7. [References](references)
8. [Licence](licence)




## Introduction

The combined approach makes it possible to generate not only a large number of syntactically correct and chemically valid molecules, but also those that have a high probability of possessing relevant chemical properties. This is done by initially modelling a large number of SMILES data with the DeepSMILESGenerator and then fine-tuning them by means of value-driven optimisation using the DeepSMILESPredictor. In addition, generated molecules can then be validated separately with the DeepSMILESPredictor for a final quality check. This method can be particularly useful in drug design to discover new, potentially effective molecules.




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

Set up all required data and parameters (these are the parameters which I have identified to achieve the best results)


- **Phase 0 DeepSMILESPredictor**

   -  You need two files as input. One file **Compound_dataset_small** containing different generic molecules enter the path under `smiles_data_path` and one file with targets **AXL_Kinase_Inhibitor_dataset** that the model should learn to predict. In this case, we want to train the model to distinguish whether a molecule is an AXL kinase inhibitor or not. Update the path under `smiles_axl_inhibitors_path`.

    Target = 1: AXL-Kinase-Inhibitor
    Target = 0: Other molecules (Compounds)

> [!TIP] 
You should search for as much target SMILES for your prediction class input as possible. In my example I found a total of 4564 on ChEMBL and NIH. For the generic smiles data I took a ratio of around 1:2 so a total of 10812 random non target SMILES and added some **class imbalance** techniques to compensate. 


   - Once the network has been trained, it will be saved under `save_model_path`. The trained SMILES Predictor model can then be used to classify SMILES.


```bash
SMILES_DATA_PATH = '/xxx/xxx/Compound_dataset_small.txt'
SMILES_AXL_INHIBITORS_PATH = '/xxx/xxx/AXL_Kinase_Inhibitor_dataset.txt'
SAVE_MODEL_PATH = '/xxx/xxx/model_predictor_0708.pth'
BATCH_SIZE = 64
LEARNING_RATE = 0.00005
NUM_EPOCHS = 150
EARLY_STOPPING_ACTIVE = True
TARGET_ACCURACY_CHEMBL = 97.2
TARGET_ACCURACY_AXL = 99.9
AXL_MULTIPLIER = 6  
COMPOUND_MULTIPLIER = 3
```



- **DeepSMILESGenerator Phase 1 base-training**

   - You need the input **Compound_dataset.txt** containing the 1.8 million small molecules compounds from the ChEMBL database. This is the dataset what I used to lead the DeepSMILESGenerator to train the vocabular and the basic structure of molecules. It is already in the repository. If you want to use a different dataset for training feel free to change it in your project and update the `file_path`. 
   
> [!TIP]
 The defined vocabulary is adjusted to the components of the molecules that occur most frequently in this data set. I have written and added a script `Vocabulary_Creator` that analyses other datasets and outputs which vocabulars are most common. With this information you can extend the vocabulary if necessary. 

   - Once the network has been pre-trained on the basic dataset, it will be saved under the `save_model_path` and can then be further trained for Transfer Learning (Phase 2).



```bash
LEARNING_RATE = 0.0010
BATCH_SIZE = 256
EPOCHS = 25
AUGMENT = True
AUGMENT_FACTOR = 5
USE_PRETRAINED_MODEL = False
LOAD_MODEL_PATH = '-'
SAVE_MODEL_PATH = '/xxx/xxx/model_generator_phase1_0708.pth'
FILE_PATH = '/xxx/xxx/Compound_dataset.txt'
ACTIVATE_FINE_TUNING = False
```



- **DeepSMILESGenerator Phase 2 pre-training**

   - First you need the pre-trained DeepSMILESGenerator model from phase 1, simply enter the path under `load_model_path` and activate `use_pretrained_model = True`. Adjust the parameters, define the second / new dataset as new `file_path` use the **AXL_Kinase_Inhibitor_dataset** for the specialization of the modell to learn generating better AXL-Kinase-Inhibitors and run the code again.


```bash
LEARNING_RATE = 0.00010
BATCH_SIZE = 256
EPOCHS = 10
AUGMENT = True
AUGMENT_FACTOR = 2
USE_PRETRAINED_MODEL = TRUE
LOAD_MODEL_PATH = '/xxx/xxx/model_generator_phase1_0708.pth'
SAVE_MODEL_PATH = '/xxx/xxx/model_generator_phase2_0708.pth'
FILE_PATH = '/xxx/xxx/AXL_Kinase_Inhibitor_dataset.txt'
ACTIVATE_FINE_TUNING = False
```



- **DeepSMILESGenerator Phase 3 fine-tuning**


   - First you need the pretrained DeepSMILESGenerator model from phase 2, simply enter the path under `trained_lstm_path` in the second part of the code.

   - Then you need the pretrained DeepSMILESPredictor model which you trained in phase 0 for classification. Simply enter the path under `trained_mlp_path` in the second part of the code.

   - Once the network has been fully trained, it will be saved under `save_model_path` in the second part of the code. The fully trained DeepSMILESGenerator model can then be used to generate specific SMILES.


```bash
TRAINED_LSTM_PATH = '/xxx/xxx/model_generator_phase2_0708.pth'
TRAINED_MLP_PATH = '/xxx/xxx/model_predictor_0708.pth'
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LEARNING_RATE = 0.00002
FINE_TUNE_BATCH_SIZE = 32
FINE_TUNE_SAVE_MODEL_PATH = '/xxx/xxx/model_generator_phase3_0708.pth'
NUM_GENERATED_SMILES = 400
REWARD_SCALE = 1
```




## Training Structure

This will give you a overview of the **DeepSMILESGenerator** and **DeepSMILESPredictor** training structure, all existing scripts in the repository and all required files:

![Project Training Structure](https://github.com/Schockwav3/SMILESGeneratorPredictor/blob/main/Pictures/project_structureV3.png)




## Project Workflow

This will give you a complete overview of the **DeepSMILESGenerator** and **DeepSMILESPredictor** Workflow:

<img src="https://github.com/Schockwav3/SMILESGeneratorPredictor/blob/main/Pictures/workflowV3.png" width="550" height="1180">





## Train DeepSMILESPredictor using a MLP and ECFPs, Molecule Descriptors for validation


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

- **The following descriptors are calculated for each molecule:**

    - **ECFPs**: A type of molecular fingerprint used for similarity searching, based on the circular substructures around each atom, and encoded as binary vectors. Extended-Connectivity Morgan Fingerprints are particularly useful in capturing the local environment of atoms within a molecule. They are generated by considering circular neighborhoods of varying radius around each atom, and then hashing these neighborhoods into fixed-length binary vectors.
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
- smiles.txt:

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

5. Processing of the ECFPs:

    - `fingerprints = np.array([list(fp) for fp in df["MorganFingerprint"].values])`: Extracts the fingerprints from the DataFrame `df` and converts them into a numpy array fingerprints. Each fingerprint is converted into a list of bits.

6. Merge the main DataFrame with the fingerprint data:

    - `df = df.drop(columns=["MorganFingerprint"])`: Removes the original MorganFingerprint column from `df` as the fingerprint data is moved to `fingerprints_df`.

    - `df = pd.concat([df, fingerprints_df], axis=1)`: Adds `fingerprints_df` to the main DataFrame `df` by appending the fingerprint columns to the main DataFrame.

- The resulting DataFrame df contains all calculated molecular descriptors, including the extracted fingerprints, for each molecule in the two SMILES datasets.




### Define the Dataset

- Process of splitting the data into training and test sets, converting this data into PyTorch tensors and creating DataLoaders to efficiently process the data during training

- In many real data sets, the classes are often unevenly distributed (**class imbalance**). This can cause problems in a binary classification problem. Machine learning models tend to favour the majority class as it occurs more frequently. This leads to poorer recognition of the minority class. By increasing the number of minority class examples, the model is forced to focus on this class as well and not just favour the majority class.

    - `axl_data = df[df['Target'] == 1]`: A subset of the DataFrame `df` is created, which only contains the data rows in which the Target column has the value `1`. This means that axl_data only represents the data points that are classified as AXL kinase inhibitors.

    - `df = pd.concat([df, axl_data] * 3)`: Duplicates the data from `axl_data` three times and inserts it back into the original DataFrame `df`.

- `X = df.drop(columns=["SMILES", "Target"])`: Removes the columns "SMILES" and "Target" from the DataFrame `df` to create the feature matrix `X`. This matrix contains all numerical descriptors and fingerprints that serve as input data for the model.

- `y = df["Target"]`: Extracts the target variable `y` from the DataFrame, which indicates whether the molecules are general molecules (`0`) or AXL kinase inhibitors (`1`).

- `train_test_split(X, y, test_size=0.2, random_state=42)`: Splits the data into training and test sets. `test_size=0.2` means that 20% of the data is used as test data. `random_state=42` ensures that the division is reproducible.

- Converts the Pandas DataFrames `X_train`, `y_train`, `X_test`, and `y_test` into PyTorch tensors.

- DataLoader for train and test data will be created


```bash
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
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

1. **Input layer:** The input layer receives vectors of molecular descriptors, the number of which is determined by the number of features in the training data.

2. **Hidden layers:** There are a total of five fully connected dense hidden layers in the architecture:

    - **Activation function:** a ReLU (Rectified Linear Unit) is applied to perform non-linear transformations.

    - **Batch normalisation:** A batch normalisation layer is inserted after each fully connected layer. These layers normalise the outputs of the previous layers, which can improve the stability and speed of the training process.

    - **Dropout:** This is a regularisation technique in which a certain rate of neurons is randomly deactivated during training. This helps to prevent overfitting.

3.  **Output layer:** The final output layer of the MLP consists of 2 neurons. These represent the two classes of the classification problem (e.g. AXL kinase inhibitor and non-inhibitor). This layer outputs raw scores (`logits`), which can then be used to calculate probabilities for the classes.

> [!TIP]
You can adjust the layer-size or add more layers to the MLP if you want to improve the results on your own dataset.




### Class Weights

- `class_weights`: A tensor that contains the weights for the classes in the training data. These are used to equalise the imbalance between the classes (**class imbalance**). The positive examples are given a higher weight to compensate for their relatively lower occurrence compared to negative examples.

- `focal_CE_loss`: This function calculates the focal loss, a modified version of the cross-entropy loss function. It focusses the training on examples that are difficult to classify.

    - `scores`: The predictions of the model (raw scores or logits before applying the softmax function).

    - `labels`: The actual labels of the data.

    - `gamma`: A hyperparameter that controls the influence of the examples that are difficult to classify. Higher values of gamma increase the focus on these examples.


```bash
criterion = F.cross_entropy
criterion = focal_CE_loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

- `optimizer = optim.Adam(model.parameters())`: Defines the optimiser that updates the weights of the model based on the calculated gradients. The **Adam optimiser** is used here

- Key Input Parameters:

    - `lr (int)`: Sets the learning rate to a small value to ensure stable and slow weight updates, which is particularly important to avoid overfitting to the training data here.




### Early Stopping

- A technique used to stop model training when certain criteria are met. In this case, training is stopped as soon as the specified accuracy targets for both classes (**CHEMBL** and **AXL**) in the test dataset are met. This ensures that the model performs well for both the majority class (**CHEMBL**) and the minority class (**AXL**), which is particularly important when the accurate classification of the minority class is of high importance.

```bash
early_stopping_active = EARLY_STOPPING_ACTIVE 
target_accuracy_chembl = TARGET_ACCURACY_CHEMBL
target_accuracy_axl = TARGET_ACCURACY_AXL
```

- Key Input Parameters:

    - `early_stopping_active (boolean)`: If early stopping is active, the training of the model is terminated prematurely as soon as the predefined conditions are reached.

    - `target_accuracy_chembl (int)`: This variable defines the target accuracy (in per cent) for the CHEMBL class (**target = 0**).

    - `target_accuracy_axl (int)`: This variable defines the target accuracy (in per cent) for the class AXL kinase inhibitors **(target = 1)**. This class is the minority class in this classification problem. A higher target accuracy for this class reflects the importance of a precise classification for this particular group. 




### Define the Trainer

- The trainer carries out the complete training of the model, monitors the performance and saves the results for later analysis and visualisation.


```bash
num_epochs = NUM_EPOCHS
```

- Key Input Parameters:

    - `num_epochs (int)`: Number of maximum training runs.
     
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




### Extract Feature Importance

- This function calculates the importance of features based on the weights of the first layer of a neural network. The idea behind this is that larger absolute weights indicate a greater importance of the corresponding feature for the predictions of the model.

1 **Extract the weights**: The function extracts the weights of the first layer (`fc1`) of the model and calculates the mean absolute weights over all neurons.

2 **Filtering the features**: If a `prefix` is specified, only the features whose names start with this prefix are considered. Otherwise, features beginning with "FP_" are excluded (typical for fingerprints).

3 **Sort by importance**: Features are sorted by their mean absolute weights.

4 **Select top features**: If `top_n` is specified, only the most important `top_n` features are returned.

5 **Output**: The function returns the names and weights of the most important features.




### Save Model


```bash
save_model_path = SAVE_MODEL_PATH
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
def plot_confusion_matrix(mdl, data, class_names=None, device=torch.device("cpu")):

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
    fig, _ = cm.plot(labels=class_names)
    return fig
```

```bash
def plot_feature_importance(top_features, top_weights, title):
    plt.figure(figsize=(14, 10))
    plt.barh(range(len(top_features)), top_weights[::-1], align='center')
    plt.yticks(range(len(top_features)), top_features[::-1], fontsize=8)
    plt.xlabel('Mean Absolute Weight')
    plt.title(title)
    plt.grid(True)
    plt.show()
```

- `plot_losses`: Visualises the progress in loss during training.

- `plot_label_accuracy`: Visualises the accuracy for different labels over the training time.

- `plot_confusion_matrix`: Visualises a confuion matrix for the predictions of a model compared to the actual labels. A confuion matrix is an important tool for evaluating the classification performance of a model as it shows the number of correctly and incorrectly classified examples for each class.

- `plot_feature_importance`: Visualises the most important features (descriptors) of the neural network. Helps to understand which descriptors make the greatest relative contribution to the classification.





## Train DeepSMILESGenerator using DeepSMILES LSTM and Reinforcement Learning with DeepSMILESPredictor


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

- The input parameters allow users to configure the model according to the complexity of the dataset and the computational resources available. The model's capability to load pretrained weights also facilitates fine-tuning (phase 3) and pre-training via Transfer Learning (phase 2), making it adaptable to new tasks with minimal retraining.


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

- The SmilesTrainer class is used to streamline the training process of the LSTM model. It handles data loading, model training, validation, and testing, as well as monitoring the training progress through loss and accuracy metrics. The plotting functions provide a clear visualization of the training dynamics, allowing for easy identification of issues such as overfitting or underfitting.

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
                 epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, use_pretrained_model=USE_PRETRAINED_MODEL, load_model_path=LOAD_MODEL_PATH, save_model_path=SAVE_MODEL_PATH):
```

- Key Input Parameters:

    - `epochs (int)`: The number of epochs (full passes through the training dataset) to train the model.

    - `learning_rate (float)`: The learning rate for the optimizer, controlling how much to adjust the model's weights with respect to the loss gradient.

    - `batch_size (int)`: The number of samples per batch to be loaded by the DataLoader.

    - `use_pretrained_model (bool)`: If this variable is set to `True`, the model is not trained from scratch, it indicates that a base-trained model should be used. It loads and uses weights that have already been learnt. Can be used for Transfer Learning or the fine-tuning process.

    - `load_model_path (str)`: Specifies the file path where the pre-trained saved model is located.

    - `save_model_path (str)`: This path specifies where the model is to be saved after base-training or pre-training. It should always be filled with a valid path otherwise the model won`t be saved.

- These settings determine whether an already trained model should be used and where this model should be saved or loaded. This is useful for the application of Transfer Learning, where a model is based on previously learnt weights to reduce training time and improve performance. Or atleast to save the base-trained model.



### Define the Dataset

- The SMILESDataset class is designed to preprocess SMILES data for the machine learning task. It efficiently handles data augmentation, tokenization, and encoding to prepare input-target pairs for model training. 

- **Data Augmentation** enhances the diversity of the dataset, potentially improving model robustness. For each SMILES in the list, if `augment=True`, it generates multiple random valid SMILES representations of the same molecule using `randomize_smiles(self, smile: str, num_random: int)` method. The model can learn to recognize the molecule regardless of the specific SMILES string used.

- `load_data`: reads a file containing SMILES strings (one per line) and returns a list of these strings.

- `split_data`: splits a given dataset into three subsets: **training=0.7**, **validation=0.15**, and **test=0.15** sets.


```bash
def __init__(self, smiles_list, tokenizer, vocabulary, augment=AUGMENT, augment_factor=AUGMENT_FACTOR):
```

- Key Input Parameters:

    - `augment (bool)`: A flag indicating whether to apply data augmentation to the SMILES strings.

    - `augment_factor (int)`: The number of augmented versions to generate for each SMILES string.




### File Path


```bash
file_path = FILE_PATH
```

- `file_path (str)`: Path to the SMILES file which includes the dataset for pre-learning of the DeepSMILESGenerator model. 

- If Transfer Learning is activated, update the path to the new SMILES file which the base-trained model should retrain. 




### Activate_fine_tuning (Phase 3)

- Control variable for activating fine-tuning (Part 3)

- `activate_fine_tuning = False` if you want to base-train the DeepSMILESGenerator model (Part 1) or pre-train with Transfer Learning (Part 2) and skip the second part of the code
- `activate_fine_tuning = True` if you want to enter fine-tuning (Part 3)


## Phase 3


### Define pre-trained Models

- Provide the paths to the saved pre-trained LSTM model which will be now be used for further training (fine-tuning) and the already trained DeepSMILESPredictor that will be used for the evaluation of the reinforcement learning rewards.

```bash
trained_lstm_path = TRAINED_LSTM_PATH
trained_mlp_path = TRAINED_MLP_PATH
```

- `trained_lstm_path (str)`: File path to the pre-trained LSTM model

- `trained_mlp_path (str)`: Path to the trained MLP model. The MLP model is used here as a predictor to evaluate the quality of the sequences generated by the LSTM model

> [!TIP]
If the DeepSMILESPredictor isn't ready pause here and continue later after training the MLP watch workflow




### MLP Definition & Descriptors

- `class MLP(nn.Module)`: Defines the multilayer perceptron (MLP) from the DeepSMILESPredictor. The output is a two-class classification. **It must be the same model architecture as in the loaded SMILES predictor mlp model**.

- `def calculate_descriptors(smiles)`: Converts a SMILES string into an RDKit molecule object and calculates a variety of chemical descriptors. **They must be the same descriptors as in the loaded SMILES predictor mlp model.**

- `def evaluate_smiles(smiles, mlp_model, tokenizer, vocabulary)`: Evaluates them with the MLP and returns the prediction. The function returns either **1 (AXL-classified)** or **0 (not AXL-classified)**, depending on the result of the MLP prediction.




### Define the Phase 3 Model Structure

- `vocabulary = FixedVocabulary()`: The vocabulary is initialised. The vocabulary from the base-training (phase 1) is used.

- `tokenizer = DeepSMILESTokenizer(vocabulary)`: The tokeniser is initialised. The tokaniser from the base-training (phase 1) is used.

- `smiles_lstm_model = SmilesLSTM(vocabulary, tokenizer)`: An instance of the SmilesLSTM model is created. The same LSTM model with the same structure from phase 1 is used. This model uses the initiated vocabulary and the tokeniser.

- `smiles_lstm_model.load_pretrained_model(trained_lstm_path)`: The pre-trained model is loaded. The previously trained weights and parameters of the model are loaded into the current instance.

- `input_size = 2048 + 46`: The input size for the MLP model is defined. It is made up of the features of the ECFPs (2048 bits) and other molecular descriptors (46 features).

- `mlp_model = load_mlp_model(trained_mlp_path, input_size)`: The DeepSMILESPredictor model is loaded into the current instance. The model is set on the defined mlp and on the `input_size`.




### Define The Trainer Phase 3

- This class implements the fine-tuning of the DeepSMILESGenerator model with a reinforcement learning approach based on classification by the DeepSMILESPredictor model.

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
class SmilesTrainerPhase2:
    def __init__(self, model, validator_model, num_generated_smiles=NUM_GENERATED_SMILES, epochs=FINE_TUNE_EPOCHS, learning_rate=FINE_TUNE_LEARNING_RATE, batch_size=FINE_TUNE_BATCH_SIZE, save_model_path=FINE_TUNE_SAVE_MODEL_PATH, reward_scale=REWARD_SCALE): 
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
    def plot_accuracy(train_accuracy, total_accuracy):
        epochs = range(1, len(train_accuracy) + 1)
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
    def plot_generated_smiles(generated_smiles_counts, axl_classified_counts):
        epochs = range(1, len(generated_smiles_counts) + 1)
        plt.figure(figsize=(14, 6))
        plt.plot(epochs, generated_smiles_counts, label='Generated SMILES')
        plt.plot(epochs, axl_classified_counts, label='AXL Classified SMILES')
        plt.xlabel('Epochs')
        plt.ylabel('Count')
        plt.title('Generated SMILES and AXL Classified SMILES Count')
        plt.legend()
        plt.grid(True)
        plt.show()
```

- `plot_accuracy`: Visualises the accuracy of the model during training. The training accuracy and the overall accuracy are visualised.

- `plot_generated_smiles`: Visualises the ratio of generated SMILES and AXL-classified SMILES across the epochs.





## Master Thesis

For a comprehensive understanding of the methodologies, analyses, and findings, please refer to the written Master’s Thesis. The study provides an in-depth exploration of each aspect, offering detailed explanations, context, and supporting data. This thorough documentation aims to ensure clarity and accessibility for readers seeking to understand the approach and conclusions of the research.




## References

- PyTorch: An open-source machine learning library.
- RDKit: Open-source cheminformatics software.
- DeepSMILES: An alternative compact representation of SMILES.
- ChEMBL Database: A comprehensive database containing bioactive molecule information.
- PubChem Database: A large, publicly accessible database providing detailed information on chemical compounds.


- **Dominik, Andreas**. Support and resources provided during the development of the project. Technical University of Applied Sciences (THM), Gießen, Germany.




## Licence

This project is intended for research and personal use. For any other inquiries or help, please contact me at Schockwav3@posteo.de
