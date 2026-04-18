# ðŸ—žï¸  Fake News Detection using SVM

A performant, clean, and modern web application built with **Python**, **Scikit-learn**, and **Streamlit** to detect fake news articles using Natural Language Processing (NLP) and Support Vector Machines (SVM).

## ðŸ’¡ Project Overview
In an era of rapid information spread, distinguishing between authentic news and misinformation is critical. This project implements a machine learning pipeline that:
1.  **Preprocesses** raw news text (cleaning, tokenization, normalization).
2.  **Extracts features** using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
3.  **Classifies** the content using a Linear Support Vector Machine (LinearSVC) model.
4.  **Provides a UI** through a stunning Streamlit web interface for real-time predictions.

## âœ¨ Features
- **High Accuracy**: Achieving **~93.4% accuracy** on the provided dataset.
- **Modern UI**: Custom-styled Streamlit interface with a clean, "stunning" aesthetic.
- **Fast Inference**: Lightweight SVM model for near-instant classification.
- **Devcontainer Support**: Fully configured for VS Code Dev Containers to ensure a consistent development environment.

## ðŸ“‚ Project Structure
```text
fakenewsdetection-gemini/
â”œâ”€â”€ .devcontainer/     # Docker & VS Code container config
â”œâ”€â”€ data/               # Raw and processed datasets (data.csv)
â”œâ”€â”€ docs/               # Project documentation (PDF/Docx)
â”œâ”€â”€ models/             # Saved model binaries (.pkl)
â”œâ”€â”€ notebooks/          # Training scripts and exploration
â”œâ”€â”€ src/                # Streamlit web application source
â”œâ”€â”€ .gitignore          # Git exclusion rules
â”œâ”€â”€ README.md           # Project documentation
â””â”€â”€ requirements.txt     # Python dependencies
```

## ðŸš€ Getting Started

### Prerequisites
- Python 3.11+ (or use the provided **Devcontainer**)
- Docker (if using Devcontainer)

### Option 1: Using VS Code Dev Containers (Recommended)
1. Open the project folder in VS Code.
2. When prompted, click **"Reopen in Container"**.
3. All dependencies and tools will be automatically installed.

### Option 2: Local Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ðŸ’» Usage

### 1. Training the Model
If the models in the `models/` folder are missing or you want to retrain:
```bash
python notebooks/train_model.py
```

### 2. Running the Web App
Launch the Streamlit interface:
```bash
streamlit run src/app.py
```
The app will be available at `http://localhost:8501`.

## ðŸ› ï¸  Technologies Used
- **Language**: Python
- **Machine Learning**: Scikit-learn (SVM, TF-IDF)
- **Data Manipulation**: Pandas, Numpy
- **Visualization**: Seaborn, Matplotlib
- **Web Framework**: Streamlit
- **Deployment**: Docker (Devcontainer)

## ðŸ“„ License
This project is developed as part of a BCA submission. See documentation in `docs/` for more details.
