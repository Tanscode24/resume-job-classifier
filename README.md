
#Resume-Job Classifier

This project is an Information Retrieval (IR) based system that matches **resumes** to the most relevant **job descriptions**. It uses **text preprocessing**, **TF-IDF vectorization**, and **cosine similarity** to rank job postings based on how well they match a given resume.

---

##Features

- Resume and job description matching
- Text preprocessing: tokenization, stopword removal, lemmatization
- Vectorization using TF-IDF
- Similarity scoring using cosine similarity
- Outputs top-N job recommendations for a given resume

---

###  Project Structure

```text
resume-job-classifier/
│
├── dataset/                          # Folder containing input data files
│   ├── jobs.csv                      # Dataset with job descriptions
│   └── resumes.csv                   # Dataset with resume texts
│
├── images/                           # Project assets like diagrams or icons
│   └── project_structure_diagram.png # Visual tree of the project structure
│
├── models/                           # Folder for saving ML models (if any)
│
├── templates/                        # HTML templates for web app (Flask/Streamlit)
│
├── utils/                            # Folder for utility and helper scripts
│   └── preprocess.py                 # Script for text cleaning and NLP preprocessing
│
├── app.py                            # Web application script (entry point for GUI)
├── ResumeCatogorization.py           # Core logic for matching resumes to jobs
├── requirements.txt                  # List of Python dependencies
└── README.md                         # Project documentation
```
### Project Structure (Visual)

The overall structure of the project is shown below:

![Project Structure](images/project_structure_diagram.png.png)
---


> 📌 **Visual Representation:**
>
> ![Project Structure](images/project_structure_diagram.png.png)

---

## ⚙️ How It Works

1. 📥 Load resume files (PDF or TXT)
2. 🔠 Preprocess with tokenization & TF-IDF
3. 🌲 Apply trained Random Forest classifier
4. 📤 Return top job role predictions

---

## 🛠️ Installation

```bash

git clone https://github.com/Tanscode24/resume-job-classifier.git
cd resume-job-classifier
pip install -r requirements.txt




