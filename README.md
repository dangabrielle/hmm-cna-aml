# hmm-cna-aml

Utilizing a Hidden Markov Model to smooth Copy Number Alterations (CNA) of the Immunoglobulin Heavy Variable V3-20 (IGHV3-20) region in Acute Myeloid Leukemia (AML)

## Installation

### Requirements

- Python 3.12 or higher (recommended: Python 3.13)
- pip package manager

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/dangabrielle/hmm-cna-aml.git
   cd hmm-cna-aml
   ```

2. **Check your Python version**

   ```bash
   python3 --version
   ```

   If you have Python 3.11 or lower, you'll need to upgrade. On macOS with Homebrew:

   ```bash
   brew install python@3.13
   ```

3. **Create a virtual environment**

   ```bash
   # Remove old venv if it exists
   rm -rf venv

   # Create new venv with Python 3.12+
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # venv\Scripts\activate   # On Windows
   ```

4. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The main dependencies:

   - `pandas` - Data manipulation and analysis
   - `numpy` - Numerical computing
   - `matplotlib` - Data visualization
   - `biopython` - Bioinformatics tools (for NCBI E-utilities)
   - `pomegranate` - Probabilistic models (HMM)

5. **Run program**

   ```bash
   python3 hmm_viterbi_aml.py
   ```

### Troubleshooting

**llvmlite build errors (cmake/compilation failures)**

This happens when using Python 3.11 or lower, which lacks pre-built wheels for `llvmlite`. Solution: upgrade to Python 3.12+.

```bash
# Check your Python version
python3 --version

# If < 3.12, upgrade (macOS with Homebrew)
brew install python@3.13

# Then recreate your venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
