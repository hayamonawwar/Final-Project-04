# Final-Project-X

## Project Overview
This repository contains the final project for our deep learning course. The project, **AI-Powered Handwritten Formula Recognition**, aims to compare the performance of **TrOCR (Transformer-based OCR)** and **CRNN (Convolutional Recurrent Neural Network)** for recognizing handwritten engineering formulas.

## Repository Structure
The repository follows the required structure:

```
Final-Project-X/
│-- Proposal/
│   ├── project_proposal.pdf
│
│-- Final-Group-Project-Report/
│   ├── group_report.pdf
│
│-- Final Presentation/
│   ├── presentation.pdf
│
│-- Code/
│   ├── train_trocr.py
│   ├── train_crnn.py
│   ├── preprocess_data.py
│   ├── evaluate_models.py
│   ├── requirements.txt
│   ├── README.md  # This file
│
│-- firstname-lastname-individual-project/
│   ├── Individual-Final-Project-Report/
│   │   ├── individual_report.pdf
│   ├── Code/
│   │   ├── my_contributions.py
```

## Getting Started

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/Final-Project-X.git
cd Final-Project-X
```

### 2. Install Dependencies
Create a virtual environment (optional but recommended):
```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
pip install -r Code/requirements.txt
```

### 3. Running the Models

#### Train TrOCR
```sh
python Code/train_trocr.py --dataset crohme --epochs 50
```

#### Train CRNN
```sh
python Code/train_crnn.py --dataset crohme --epochs 50
```

#### Evaluate Models
```sh
python Code/evaluate_models.py --metric cer
```

## Team Members
- **[Your Name]**
- **[Teammate 1]**
- **[Teammate 2]**

## Project Description
This project explores **deep learning-based handwritten formula recognition**, comparing **TrOCR** and **CRNN** architectures. We fine-tune **TrOCR using Hugging Face** and train **CRNN with CTC loss**, evaluating their **character error rate (CER), word error rate (WER), and BLEU score**.

## Contact
For any questions or issues, please reach out via GitHub Issues or email **[your-email]**.

---

🛠 **Maintained by:** [Your Name]  
📅 **Project Start Date:** March 2025  
📂 **Course:** Deep Learning (Spring 2025)
