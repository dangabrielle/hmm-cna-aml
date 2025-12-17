# hmm-cna-aml

Utilizing a Hidden Markov Model to smooth Copy Number Alterations (CNA) of the Immunoglobulin Heavy Variable V3-20 (IGHV3-20) region in Acute Myeloid Leukemia (AML)

## Installation

### Requirements

- Python 3.7 or higher
- pip package manager

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/dangabrielle/hmm-cna-aml.git
   cd hmm-cna-aml
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # for Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies are:

   - `pandas` - Data manipulation and analysis
   - `numpy` - Numerical computing
   - `matplotlib` - Data visualization
   - `biopython` - Bioinformatics tools (for NCBI E-utilities)

4. **Configure NCBI Email** (required for Entrez queries):

   - Open `hmm_viterbi_aml.py`
   - Replace `"your.email@example.com"` with your actual email address on line 15:
     ```python
     Entrez.email = "your.email@example.com"  # Replace with your actual email
     ```

5. **Verify installation**:
   ```bash
   python3 hmm_viterbi_aml.py
   ```

### Troubleshooting

- If you encounter import errors, ensure all dependencies are installed: `pip install --upgrade pandas numpy matplotlib biopython`
- For NCBI access issues, verify your email is correctly set in the script
- On macOS/Linux, ensure you have Python 3 installed: `python3 --version`
