# Handwriting-Style-Synthesis
Implementation of "A Harmonic Approach to Handwriting Style Synthesis Using Deep Learning." Features a Style Encoder, Text Generator, and Dual Discriminators to produce diverse, realistic handwriting, including out-of-vocabulary text. Applications span digital forensics, creative writing, and document security.
# A Harmonic Approach to Handwriting Style Synthesis Using Deep Learning

## Overview

This repository contains the official implementation of the paper:

**"A Harmonic Approach to Handwriting Style Synthesis Using Deep Learning"**  
Mahatir Ahmed Tusher, Saket Choudary Kongara, Sagar Dhanraj Pande, SeongKi Kim, Salil Bharany  
Published in *Computers, Materials & Continua*, 2024.  
[DOI: 10.32604/cmc.2024.049007](https://doi.org/10.32604/cmc.2024.049007)

The challenging task of handwriting style synthesis requires capturing the individuality and diversity of human handwriting. This project presents a novel deep learning model that combines a **Style Encoder**, **Text Generator**, and **Dual Discriminators** to generate diverse, realistic, and out-of-vocabulary handwriting styles. By focusing on the synthesis of personalized and visually appealing handwriting, the proposed approach significantly advances the field.
![image](https://github.com/user-attachments/assets/c01c7f64-4ca4-4f2e-a78a-23c7b7c1b0ff)
**Figure:** Architecture overview of the proposed approach's framework along with the generation of the word "danger".

### Key Features:
- **Style Encoder**: Extracts style vectors from handwriting samples, encapsulating unique features of individual styles.
- **Text Generator**: Generates high-quality, conditional text images based on the extracted style vectors.
- **Dual Discriminators**: Evaluate individual character quality and cursive joins to ensure the synthesized handwriting appears natural and coherent.
- **Performance**:
  - **Geometric Score (GS)**: `3.21 × 10⁻⁵`
  - **Fréchet Inception Distance (FID)**: `8.75`
  - Achieved competitive Word Error Rate (WER) and Character Error Rate (CER) on IAM and RIMES datasets.

### Applications:
- **Digital Forensics**: Analyze handwriting for authenticity and verification.
- **Creative Writing**: Enable authors to explore novel handwriting styles and designs.
- **Document Security**: Prevent forgery by generating unique, hard-to-replicate handwriting styles.

---

## Repository Structure

```plaintext
handwriting-synthesis/
│
├── models/
│   ├── style_encoder.py        # Style Encoder model
│   ├── text_generator.py       # Text Generator model
│   ├── char_discriminator.py   # Character-Level Discriminator
│   ├── cursive_discriminator.py # Cursive-Level Discriminator
│
├── training/
│   ├── train.py                # Training loop
│
├── datasets/
│   ├── load_data.py            # Dataset preprocessing and loading
│
├── utils/
│   ├── losses.py               # Loss functions
│   ├── metrics.py              # Evaluation metrics
│
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
```

---

## Requirements

To replicate the experiments, ensure you have the following dependencies installed:

```plaintext
tensorflow==2.12.0
numpy
matplotlib
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Dataset Preparation
The framework is trained and evaluated on two major handwriting datasets: **IAM** and **RIMES**.

### IAM Dataset  
- Download the dataset from: [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)  
- Directory partitions can be downloaded from: [IAM Laia Dataset Partition](https://github.com/jpuigcerver/Laia/tree/master/egs/iam/data/part/lines/original)

### RIMES Dataset  
- Download the dataset from: [RIMES Database](http://www.a2ialab.com/doku.php?id=rimes_database:start)

Update the file paths in `datasets/load_data.py` to ensure the datasets are correctly loaded for training.

### 2. Training the Model
To train the handwriting synthesis model, execute the following command:

```bash
python training/train.py
```

This will:
1. Extract style vectors from the handwriting samples using the **Style Encoder**.
2. Generate text images stroke by stroke using the **Text Generator**.
3. Optimize the model using **Dual Discriminators** to ensure realistic output.

### 3. Generating Handwriting Samples
Once training is complete, you can use the trained model to synthesize handwriting styles. Create a dedicated script for inference or modify the training script to evaluate and visualize results.

![image](https://github.com/user-attachments/assets/43c255d5-4c40-417a-bda2-8266da1079a4)

**Figure:** For every handwriting style, eight sentences were created using a selected reference word.

---

## Model Architecture

### 1. **Style Encoder**
- A CNN-based architecture trained to extract low-dimensional style vectors (128-dimensional) from handwriting samples.
- Encodes unique features like slants, pressure, and spacing into a compact representation.

### 2. **Text Generator**
- A modified Sketch-RNN that generates handwriting stroke by stroke, conditioned on style vectors.
- Utilizes attention mechanisms and mixture density networks to enhance stroke accuracy.

### 3. **Dual Discriminators**
- **Character Discriminator**: Ensures that individual characters in the generated handwriting are realistic.
- **Cursive Discriminator**: Evaluates and fine-tunes cursive joins to ensure smooth and natural handwriting flow.

![image](https://github.com/user-attachments/assets/80dbf200-9871-4444-ab7c-5587e91ca74c)

**Figure:** Model Architecture for Handwriting Style Synthesis with Style Encoder, Text Generator, and Dual Discriminator

---

## Results

### Evaluation Metrics:
The model achieved state-of-the-art results on the IAM and RIMES datasets:
- **Geometric Score (GS)**: `3.21 × 10⁻⁵`
- **Fréchet Inception Distance (FID)**: `8.75`
- **Word Error Rate (WER)** and **Character Error Rate (CER)** on IAM and RIMES datasets:

| Metric | IAM  | RIMES |
|--------|------|-------|
| WER    | 14.71% | 11.57% |
| CER    | 5.32%  | 3.27%  |
| FID    | 8.75   | N/A    |

### Comparison with Prior Methods:
- Outperformed previous techniques in generating diverse and realistic handwriting styles.
- Improved generalization for out-of-vocabulary words and unseen text samples.

![image](https://github.com/user-attachments/assets/eaa001fe-26a3-4f6d-b293-b8edb3b4d3eb)

  **Figure:** The comparison between the proposed method’s result with following prior studies: Luo et al. (a), Alonso et al. (b), Fogel at el. (c), and Gan et al. (d) and  (e) in the figure displays the proposed approach’s result.

### Diversity and Style Interpolation:
The framework effectively produces handwriting with diverse shapes, varying letter spacing, and adjustable stylistic features. It also excels in **style interpolation**, blending multiple styles to create smooth transitions, as shown in the examples provided in the paper.

---

## Citation

If you use this code or model in your research, please cite the paper:

```bibtex
@article{tusher2024harmonic,
  title={A Harmonic Approach to Handwriting Style Synthesis Using Deep Learning},
  author={Tusher, Mahatir Ahmed and Kongara, Saket Choudary and Pande, Sagar Dhanraj and Kim, SeongKi and Bharany, Salil},
  journal={Computers, Materials \& Continua},
  volume={79},
  number={3},
  pages={4063--4080},
  year={2024},
  publisher={Tech Science Press},
  doi={10.32604/cmc.2024.049007}
}
```

---
Download and read the paper [here.](https://doi.org/10.32604/cmc.2024.049007) 

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

---

## Funding Statement

This work was supported by the **National Research Foundation of Korea (NRF)** Grant funded by the Korean government (MSIT) (NRF-2023R1A2C1005950).

## Attention
The tool is only free for academic research purposes.

