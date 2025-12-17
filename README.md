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
   source venv/bin/activate
   ```

3. **Install required dependencies**:

   ```bash
   pip install llvmlite==0.43.0 --only-binary :all:
   pip install "numpy<2"
   pip install -r requirements.txt
   ```

   Note: Installing `llvmlite` separately with `--only-binary :all:` ensures a pre-built wheel is used, avoiding compilation issues on macOS.

   The main dependencies:

   - `pandas` - Data manipulation and analysis
   - `numpy` - Numerical computing
   - `matplotlib` - Data visualization
   - `biopython` - Bioinformatics tools (for NCBI E-utilities)
   - `pomegranate` - Probabilistic models (HMM)

4. **Configure Entrez email**:

   Open `hmm_viterbi_aml.py` and set your email address for NCBI Entrez API access:

   ```python
   Entrez.email = 'test@email.com'
   ```

5. **Run program**:
   ```bash
   python3 hmm_viterbi_aml.py
   ```
