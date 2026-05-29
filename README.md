🧠 NLP Mental Health Text Classifier – Transformer Model Comparison

MSc Dissertation project comparing Transformer-based NLP models for classifying mental health-related text — combining large-scale ETL pipeline design with state-of-the-art natural language processing.

📊 Project Overview

This project develops and compares multiple Transformer-based NLP models to classify mental health-related text data at scale. The work was completed as an MSc Data Science dissertation at the University of Surrey.
The core research question: Which Transformer architecture best identifies and categorises mental health discourse in unstructured text?

🔑 Key Contributions

Compared multiple Transformer architectures (BERT-based models) for mental health text classification
Engineered features and built ETL pipelines capable of processing large-scale unstructured text datasets
Evaluated models on accuracy, precision, recall, and F1-score
Presented findings with emphasis on real-world business and clinical decision-making applications


🛠️ Tech Stack

Tool Usage Python Core development language HuggingFace Transformers Pre-trained Transformer models (BERT, RoBERTa) pandas, NumPy Data processing and feature engineering scikit-learn Model evaluation and metrics Jupyter Notebook Experimentation and reporting

📁 Project Structure

├── data/              # Dataset loading and preprocessing

├── models/            # Transformer model training and comparison

├── pipelines/         # ETL pipelines for unstructured text

├── evaluation/        # Metrics, results, and comparison analysis

└── README.md

🚀 How to Run

bash# Clone the repo
git clone https://github.com/dhruvdedhia/Mental-Health.git
cd Mental-Health

# Install dependencies

pip install -r requirements.txt

# Run model training

python models/train.py

💡 Research Impact

Demonstrates applied NLP at scale — from raw unstructured text to structured, classified output
ETL pipeline design is transferable to any large-scale text classification task
Findings relevant to mental health tech, clinical NLP, and AI-assisted diagnosis tools


👤 Author

Dhruv Dedhia — Data Analyst | MSc Data Science, University of Surrey
