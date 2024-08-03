# SMILES Generator using DeepSMILES LSTM and Reinforcement Learning with the SMILES Predictor

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

**Conda**

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




## Requirements

- Set up all required data

- **Phase 1 SMILES Generator**

   - You need the input **chembl_smiles.txt** containing the 1.8 million small molecule compounds from the ChEMBL database. This is the dataset what I used to lead the SMILES Predictor to train the vocabular and the basic structure of molecules. It is already in the repository. If you want to use a different dataset for training feel free to change it in your project and update the `file_path`. 
   
> Note: The defined vocabulary is adjusted to the components of the molecules that occur most frequently in this data set. I have written and added a script `Vocabulary_Creator` that analyses other datasets and outputs which vocabulars are most common. With this information you can extend the vocabulary if necessary. 

   - Once the network has been pretrained on the basic dataset, it will be saved under the `save_model_path` and can then be further trained for Transfer Learning or phase 2.


> Note: It is also possible to use **Transfer Learning** to a second more specific dataset of SMILES. To do this, simply save the model after the first training and activate `use_pretrained_model = True`, adjust the parameters, define the second / new dataset as new `file_path` and run the code again.


- **Phase 2 SMILES Generator**

   - First you need the pretrained SMILES Generator model from phase 1, simply enter the path under `trained_lstm_path` in the second part of the code.

   - Then you need the pretrained SMILES Predictor model which you will also need in phase 3 for classification. Simply pause here and prepare the learning for the predictor model. Afterwards simply enter the path under `trained_mlp_path` in the second part of the code.

   - Once the network has been finally trained, it will be saved under `save_model_path` in the second part of the code. The fully trained SMILES Generator model can then be used to generate specific SMILES.


- **Phase 3 SMILES Predictor**

   -  You need two files as input. One file **smiles_data** containing different generic molecules enter the path under `smiles_data_path` and one file with targets **smiles_axl_inhibitors** that the model should learn to predict. In this case, we want to train the model to distinguish whether a molecule is an AXL kinase inhibitor or not. Update the path under `smiles_axl_inhibitors_path`.

    Target = 1: AXL kinase inhibitor
    Target = 0: Other molecules

> Note: You should search for as much target SMILES for your prediction class input as possible. In my example I found a total of 4564 on ChEMBL and NIH. For the generic smiles data I took a ratio of around 1:2 so a total of 10812 random non target SMILES. 


   - Once the network has been trained, it will be saved under `save_model_path`. The trained SMILES Predictor model can then be used to classify SMILES.




## Project Structure

This will give you a complete overview of the **SMILES GeneratorPredictor** project structure, all existing scripts in the repository and all required files:

![Project Structure](https://github.com/Schockwav3/SMILESGeneratorPredictor/blob/main/Pictures/project_structure.png)




## Workflow

This will give you a complete overview of the **SMILES GeneratorPredictor** Workflow:

<img src="https://github.com/Schockwav3/SMILESGeneratorPredictor/blob/Pictures/main/workflow.png" width="600" height="1360">








4. **SMILES Generator using DeepSMILES LSTM and Reinforcement Learning with the SMILES Predictor**
    1. [Data Preparation](#data-preparation)
    2. [Model Training](#model-training)
    3. [Siamese Network Training](#siamese-network-training)
    4. [Combining Both Models](#combining-both-models)
    5. [Usage](#usage)
    6. [Parameters](#parameters)

5. **SMILES Predictor using a MLP and Morgan Fingerprints, Molecule Descriptors for classification**
    1. [Data Preparation](#data-preparation)
    2. [Model Training](#model-training)
    3. [Siamese Network Training](#siamese-network-training)
    4. [Combining Both Models](#combining-both-models)
    5. [Usage](#usage)
    6. [Parameters](#parameters)

6. [References](#references)







## Requirements

- python 3.7+
- rdkit
- pyTorch
- numPy
- pandas
- scikit-learn
- tqdm
- matplotlib
- torchmetrics
- deepsmiles


## Project Structure

├── data
│ ├── chembl_smiles.txt
│ ├── smiles_example3.txt
│ └── ChemblDatensatzAXLKinase.txt
├── src
│ ├── xx.py
│ ├── xx.py
│ ├── xx.py
│ ├── xx.py
│ └── Vocabulary_Creator.ipynb
├── README.md
└── requirements.txt


## Setup and Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Schockwav3/SMILESGeneratorPredictor.git
    cd smiles-generation
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```



## SMILES Generator using DeepSMILES LSTM and Reinforcement Learning with the SMILES Predictor


## Define the Vocabulary

The FixedVocabulary class defines a fixed set of tokens for encoding SMILES sequences.

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
If you need to update or change the FixedVocabulary you can use the sript in /src/Vocabulary_Creator.ipynb to analyze a file with SMILES and see which Tokens are used and how many of them are included to create a updated Vocabulary but for most use cases this Vocabulary should be fine.


## Tokanizer

- The DeepSMILESTokenizer class handles the transformation of SMILES into deepSMILES and performs tokenization and untokenization.

- The DeepSMILESTokenizer class uses several regular expressions to tokenize deepSMILES strings. Each regex pattern is designed to  match specific components of a SMILES string. Below are the regex patterns used and their purposes:

    - `brackets` groups characters within square brackets together, which can represent charged atoms or specific configurations in the SMILES syntax.

    - `2_ring_nums` matches numbers up to two digits preceded by a percent sign ("%"), used to denote ring closures in molecules with more than 9 rings.

    - `brcl` matches the halogen atoms bromine ("Br") and chlorine ("Cl"), ensuring they are recognized as unique tokens in the SMILES string. They are essential in drug molecules.


 ```bash
       "brackets": re.compile(r"(\[[^\]]*\])"),
       "2_ring_nums": re.compile(r"(%\d{2})"),
       "brcl": re.compile(r"(Br|Cl)")
 ```


## Define the LSTM Model (RNN)



## Data Preparation

1. Prepare your SMILES data files:
    - `chembl_smiles.txt`: Contains 1.8 million molecule SMILES strings from Chembl Database for the base training of the LSTM
    - `ChemblDatensatzAXLKinase.txt`: Contains known AXL kinase inhibitors from Chembl / NIB Database for the training of the Siamse Network
    - `smiles_example3.txt`: Contains other molecule SMILES strings for the training of the Siamese Network
   

2. Place these files in the `data` directory.

## Model Training

1. Define the vocabulary and tokenizer:
    ```python
    vocabulary = FixedVocabulary()
    tokenizer = DeepSMILESTokenizer(vocabulary)
    ```

2. Initialize and train the SMILES generation model:
    ```python
    model = SmilesLSTM(vocabulary, tokenizer)
    model.to(device)

    smiles_list = load_data('data/smiles_example3.txt')
    dataset = SMILESDataset(smiles_list, tokenizer, vocabulary, augment=True)

    train_dataset, valid_dataset, test_dataset = split_data(dataset)

    train_loader = DataLoader(train_dataset, batch_size=250, shuffle=True, collate_fn=SmilesLSTM.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=250, shuffle=False, collate_fn=SmilesLSTM.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=250, shuffle=False, collate_fn=SmilesLSTM.collate_fn)

    trainer = SmilesTrainer(model, train_loader, valid_loader, test_loader, use_pretrained_model=False)
    trainer.train()
    ```

## Siamese Network Training

1. Prepare the AXL and non-AXL SMILES lists:
    ```python
    axl_smiles_list = load_data('data/ChemblDatensatzAXLKinase.txt')
    non_axl_smiles_list = load_data('data/smiles_example3.txt')
    ```

2. Initialize and train the Siamese network:
    ```python
    siamese_dataset = SiameseDataset(axl_smiles_list, non_axl_smiles_list, tokenizer, vocabulary, num_pairs=10)
    siamese_loader = DataLoader(siamese_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    siamese_base_model = SmilesLSTM(vocabulary, tokenizer)
    siamese_model = SiameseNetwork(siamese_base_model)

    train_siamese_network(siamese_model, siamese_loader, epochs=10)
    ```

## Combining Both Models

1. Initialize the combined trainer and train both models together:
    ```python
    trainer_with_siamese = SmilesTrainerWithSiamese(model, siamese_model, train_loader, valid_loader, test_loader, 
                                                    axl_smiles_list=axl_smiles_list, tokenizer=tokenizer, 
                                                    vocabulary=vocabulary, use_pretrained_model=False)
    trainer_with_siamese.train()
    ```

## Usage

- Train the models:
    ```bash
    python src/main.py
    ```

- Change the `file_path` in `main.py` to use your custom data files.

## Parameters

### SmilesTrainer
- `model`: The SMILES generation model.
- `train_dataloader`: DataLoader for training data.
- `valid_dataloader`: DataLoader for validation data.
- `test_dataloader`: DataLoader for testing data.
- `epochs`: Number of training epochs (default: 5).
- `learning_rate`: Learning rate for the optimizer (default: 0.0010).
- `batch_size`: Batch size for training (default: 250).
- `use_pretrained_model`: Whether to use a pretrained model (default: None).
- `load_model_path`: Path to the pretrained model (default: None).
- `save_model_path`: Path to save the trained model (default: None).

### SmilesTrainerWithSiamese
- `model`: The SMILES generation model.
- `siamese_model`: The Siamese network model.
- `train_dataloader`: DataLoader for training data.
- `valid_dataloader`: DataLoader for validation data.
- `test_dataloader`: DataLoader for testing data.
- `axl_smiles_list`: List of AXL kinase inhibitor SMILES.
- `tokenizer`: The DeepSMILES tokenizer.
- `vocabulary`: The vocabulary object.
- `epochs`: Number of training epochs (default: 10).
- `learning_rate`: Learning rate for the optimizer (default: 0.0010).
- `batch_size`: Batch size for training (default: 250).
- `use_pretrained_model`: Whether to use a pretrained model (default: None).
- `load_model_path`: Path to the pretrained model (default: None).
- `save_model_path`: Path to save the trained model (default: None).



## SMILES Predictor using a MLP and Morgan Fingerprints, Molecule Descriptors for validation


## Descriptors

The following descriptors are calculated for each molecule:

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


## Target Variable

The target variable indicates whether a molecule is an AXL kinase inhibitor or not:

- `Target = 1`: AXL kinase inhibitor
- `Target = 0`: Other molecules


## Steps to Create the Training Dataset

1. **Read SMILES Data**:
    - Two files are used: one (`smiles.txt`) containing various molecules, and another (`axl_inhibitors.txt`) with known AXL kinase inhibitors.

2. **Calculate Descriptors**:
    - For each molecule, various molecular descriptors are calculated.

3. **Set Target Variable**:
    - For molecules from `smiles.txt`, set `Target` to 0.
    - For molecules from `axl_inhibitors.txt`, set `Target` to 1.

## Example

Given the following data:

- `smiles.txt`:
    ```
    CCO
    CCC
    COC
    ```

- `axl_inhibitors.txt`:
    ```
    C1=CC=CC=C1
    C2=CCN=C(C)C2
    ```

The descriptors and targets are assigned as follows:

- `smiles.txt`:
    ```
    CCO: Descriptors + Target = 0
    CCC: Descriptors + Target = 0
    COC: Descriptors + Target = 0
    ```

- `axl_inhibitors.txt`:
    ```
    C1=CC=CC=C1: Descriptors + Target = 1
    C2=CCN=C(C)C2: Descriptors + Target = 1
    ```

## Goal of the Model

The model aims to utilize the descriptors to learn whether a molecule is an AXL kinase inhibitor (`Target = 1`) or not (`Target = 0`).


## Training and Validation

1. **Train-Test Split**:
    - The data is split into training and test datasets. The training set is used to train the model, and the test set is used to evaluate the model's performance.

2. **Model Architecture**:
    - A Multi-Layer Perceptron (MLP) is used to process the descriptors and learn to classify molecules as AXL kinase inhibitors or not.

3. **Training the Model**:
    - The model is trained on the training dataset, adjusting its weights to improve predictions.

4. **Validation**:
    - After training, the model is validated on the test dataset to assess its accuracy and performance.




## Results and Visualization

- The training and test losses are plotted to visualize the training progress.
- The accuracy for each label (AXL kinase inhibitors and other molecules) is plotted for both training and test datasets.
- Confusion matrices are plotted to visualize the model's performance on the training and test datasets.


## Saving and Loading the Model

- The trained model is saved to a file (`model_predictor.pth`) and can be loaded for further predictions or evaluations.

















## Results

The models generate SMILES strings based on the training data and validate their performance using known AXL kinase inhibitors. The results are plotted for loss and accuracy over epochs.

## References

- RDKit: Open-source cheminformatics software.
- DeepSMILES: An alternative compact representation of SMILES.
- PyTorch: An open-source machine learning library.




- `chembl_smiles.txt`: Contains 1.8 million molecule SMILES strings from Chembl Database for the base training of the LSTM
    - `ChemblDatensatzAXLKinase.txt`: Contains known AXL kinase inhibitors from Chembl / NIB Database for the training of the Siamse Network