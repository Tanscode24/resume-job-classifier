
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

##Project Structure
resume-job-classifier/
│
├── data/                        # Folder containing datasets
│   └── jobs.csv                 # Job descriptions dataset
│   └── resumes.csv              # Resume texts (optional or synthetic)
│
├── utils/                       # Utility functions for preprocessing
│   └── preprocess.py            # Text cleaning and NLP steps
│
├── main.py                      # Main script to run the matching pipeline
│
├── requirements.txt             # List of required Python packages
│
└── README.md                    # Project documentation (you’re reading it!)

### 📊 Project Structure (Visual)




---

## How It Works

1. **Load Data**: Import resume(s) and job descriptions from the CSV files.
2. **Preprocess**: Clean and tokenize the text (remove stopwords, lemmatize).
3. **TF-IDF Vectorization**: Represent text as numerical feature vectors.
4. **Cosine Similarity**: Compare the resume vector with each job vector.
5. **Rank & Recommend**: Output top-N most similar jobs for each resume.

---

## 🧪 Example

```bash
$ python main.py

