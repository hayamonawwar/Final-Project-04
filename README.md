# Final-Project-04

## Project Overview
This repository contains the final project for our deep learning course. The project, **AI-Powered Handwritten Formula Recognition**, aims to compare the performance of our baseline model (inspired from Kaggle competitions on the subject) and our own modified **CRNN (Convolutional Recurrent Neural Network)** for recognizing handwritten engineering formulas. The team members of this project are: Haya Monawwar and Sungjoo Chung, supervised by Dr. Martin Hagan.

## Repository Structure
The repository follows the required structure:

```
Final-Project-04
│-- Proposal
│   ├── project_proposal.pdf
│
│-- Final-Group-Project-Report
│   ├── group_report.pdf
│
│-- Final Presentation
│   ├── Final Presentation.pdf
│
│-- Code
│   ├── CRNN_HMER.ipynb # Final code/model 
│   ├── data.zip
│   ├── preprocessing.py
│   ├── README.md
│
│-- firstname-lastname-individual-project
│       ├── Individual-Final-Project-Report.pdf
│       ├── Code
```
## Training the Model
To run this HMER model, save the files in the folder 'Code' in one location. Extract the files from 'data.zip'. Since the training, testing, and validation folders are pre-made within the 'data' folder, simply executing the cells in the CRNN_HMER.ipynb file in order will run the code and train the model successfully. The code in the pre-processing file is just for reference and to detail the pre-processing steps that were used. This file can be run by using the terminal and entering the command 'python preprocessing.py'. 

The separate code files in our individual folders only focus on our individual contributions to the final code, which is compiled accurately in CRNN_HMER.ipynb file. As such, these individual code files need not be run individually.
