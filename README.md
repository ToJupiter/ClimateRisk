# Climate Risk Impact on Firm Value: A Causal Analysis using Double ML

This project analyzes the causal impact of climate-related risk disclosures in corporate annual reports on a company's firm value, measured by Tobin's Q. The analysis pipeline involves several stages: text extraction from PDF reports, TF-IDF score calculation for climate risk keywords, and a causal machine learning analysis using Double Machine Learning (DML).

## Project Overview

The primary goal is to answer the question: **Does increased discussion of climate risk in annual reports causally affect a company's financial valuation?**

To achieve this, the project implements the following workflow:

1.  **PDF to Text Conversion**: Automatically extracts raw text from a large corpus of corporate annual reports in PDF format.
2.  **Climate Risk Quantification**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to calculate a `ClimateRiskScore` for each report based on a predefined vocabulary of climate-related terms.
3.  **Causal Inference**: Employs the EconML library to build a causal model (LinearDML) that estimates the Average Treatment Effect (ATE) of the `ClimateRiskScore` on the firm's Tobin's Q, while controlling for various economic confounders.
4.  **Moderator Analysis**: Investigates the moderating effect of a company's ESG (Environmental, Social, and Governance) score on the relationship between climate risk disclosure and firm value.

## File Structure

The main components of the project are organized as follows:

-   `main.py`: The main entry point for running the analysis pipeline. It provides a command-line interface to execute different stages of the project.
-   `pdf_processor.py`: Contains functions for converting PDF files into plain text using the `PyMuPDF` library. It is optimized for performance using multiprocessing.
-   `tfidf_calculator.py`: Responsible for calculating the TF-IDF scores for climate risk keywords found in the extracted text files.
-   `causal_analysis.py`: Implements the core causal inference logic using `EconML`. It defines and runs multiple analysis cases, generates results, and creates visualizations.
-   `config.py`: A centralized configuration file for managing file paths, keywords, and other parameters.
-   `requirements.txt`: Lists all the Python dependencies required to run the project.
-   `output_txt/`: Default directory for storing the extracted text files from PDFs.
-   `output_tfidf/`: Default directory for storing the CSV file with the calculated TF-IDF scores.

## Methodology

### 1. Text Extraction

The process starts by recursively scanning a directory of PDF annual reports. The `pdf_processor.py` script reads each PDF, extracts its text content page by page, and saves it as a `.txt` file in the `output_txt` directory, preserving the original folder structure.

### 2. TF-IDF for Climate Risk

Once the text is extracted, `tfidf_calculator.py` preprocesses it by converting to lowercase and removing punctuation. It then uses `scikit-learn`'s `TfidfVectorizer` with a specific vocabulary of climate-related keywords (defined in `config.py`) to calculate a TF-IDF score for each term. The scores for all keywords are summed to create a single `ClimateRiskScore` for each document, which quantifies the extent of climate risk discussion in the report.

### 3. Causal Inference with Double Machine Learning (DML)

The core of the analysis is performed in `causal_analysis.py`. We use a Double Machine Learning model to isolate the causal effect of climate risk disclosure on firm value.

-   **Outcome Variable (Y)**: Tobin's Q (a measure of firm value).
-   **Treatment Variable (T)**: The `ClimateRiskScore` calculated in the previous step.
-   **Confounders (X)**: A set of economic variables that could influence both the treatment and the outcome, such as company size, leverage, debt ratios, GDP growth, and inflation.
-   **Moderator (W)**: The company's ESG score, used to explore how the treatment effect varies across different levels of ESG performance.

The DML model works in two stages:
1.  It trains two machine learning models (e.g., RandomForest or XGBoost) to predict the outcome and the treatment based on the confounders.
2.  It then computes the causal effect by analyzing the relationship between the residuals from these two models. This process helps to remove the confounding effects and isolate the true causal impact of the treatment.

The analysis runs four distinct cases to provide a comprehensive view:
-   **Case 1**: Simple model without confounders or moderator.
-   **Case 2**: Model with confounders.
-   **Case 3**: Model with a moderator but no confounders.
-   **Case 4**: Full model with both confounders and a moderator.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ToJupiter/ClimateRisk.git
    cd ClimateRisk
    ```

2.  **Create a virtual environment (uv recommended):**
    ```bash
    uv venv -p 3.9
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    The script will attempt to download the `stopwords` corpus from NLTK automatically. If this fails, you can run the following in a Python interpreter:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## Usage

The main script `main.py` provides a simple interface to run the different parts of the analysis.

1.  **Configure Paths**: Before running, ensure the paths in `config.py` (e.g., `NEW_DATASET_FILE`, `NEW_DATASET_OUTPUT`, `NEW_FINAL_CSV_PATH`) are set correctly for your environment.

2.  **Run the main script**:
    ```bash
    python main.py
    ```

3.  **Choose a task**:
    You will be prompted to select one of the following options:
    -   **Option 1: Converting PDFs to Text file**: Runs the PDF extraction process.
    -   **Option 2: Calculate TF-IDF scores**: Calculates TF-IDF scores from the text files in the output directory.
    -   **Option 3: Run Causal Machine Learning model**: Executes the full causal analysis and generates plots (`distribution_plots.png`, `cate_by_esg.png`, etc.).

## Dependencies

The project relies on the following major libraries:

-   `pandas` & `numpy` for data manipulation.
-   `PyMuPDF` for PDF text extraction.
-   `scikit-learn` for TF-IDF calculation and machine learning models.
-   `econml` for causal inference with Double Machine Learning.
-   `xgboost` as a machine learning model for the DML estimator.
-   `matplotlib` & `seaborn` for plotting.
-   `shap` for model interpretability and feature importance.

For a full list, see `requirements.txt`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.