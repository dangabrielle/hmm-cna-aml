#!/usr/bin/env python3
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import Entrez

"""
This script performs genomic region analysis and HMM-based CNA state prediction
on AML patient data, focusing on the IGHV3-20 gene region on chromosome 14
"""

"""
    Viterbi Algorithm Implementation using log probabilities, core algorithm
    used in this project

    Parameters:
    -----------
    y : observed sequence of CNA values ex -[2,2,1,3,3]
    A : transition matrix
    B : emission matrix
    Pi : initial probabilities

    Returns:
    --------
    x : np.ndarray
        Most likely state sequence
    T1 : np.ndarray
        Viterbi log-probability matrix
    T2 : np.ndarray
        Traceback matrix (backpointers)
"""
def viterbi(y, A, B, Pi=None):
    # gets number of hidden states (K states - model-dependent)
    K = A.shape[0]

    # accounts for lack of initial probabilities, sets them to equal probabilities if
    # none
    Pi = Pi if Pi is not None else np.full(K, 1 / K)

    # get the length of observation sequence ie the num of genes in 
    # region, in this case we're observing 7 
    T = len(y)

    # Use log probabilities to avoid numerical underflow
    # small value to make sure we prevent log(0) -> -inf, breaks 
    epsilon = 1e-10

    # convert to log space since the numbers will get smaller and smaller
    # must enforce pricision
    log_A = np.log(A + epsilon)
    log_B = np.log(B + epsilon)
    log_Pi = np.log(Pi + epsilon)

    # initialize a K x T matrix
    # rows (K) - each hidden state (model-dependent: e.g., 0=neutral, 1=gain for 2-state)
    # cols (T) - each gene
    # T1 holds log probability of being at a particular state (K) at a particular gene (T)
    T1 = np.empty((K, T), 'd')
    # stores the previous state that led to the probability in T1
    T2 = np.empty((K, T), 'B')

    # Initialization (log space)
    # init first column of T1 
    # ex:
    # log_Pi                 +  log_B[:, y[0]]
    # [-3.69, -0.05, -3.69]  +  [log(B[0,2]), log(B[1,2]), log(B[2,2])]
    T1[:, 0] = log_Pi + log_B[:, y[0]]
    T2[:, 0] = 0

    # Forward pass (log space)
    for i in range(1, T):
        for j in range(K):
            # Calculate log probabilities for transitioning to state j
            log_probs = T1[:, i - 1] + log_A[:, j] + log_B[j, y[i]]
            T1[j, i] = np.max(log_probs)
            T2[j, i] = np.argmax(log_probs)

    # Backward pass (traceback)
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])

    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2

"""
    Extract HMM parameters (A, B, Pi) from a trained pomegranate DenseHMM model.

    Parameters:
    -----------
    model : DenseHMM
        Trained pomegranate HMM model

    Returns:
    --------
    A : np.ndarray
        Transition matrix (K × K)
    B : np.ndarray
        Emission matrix (K × num_observations)
    Pi : np.ndarray
        Initial state probabilities (K,)
    """
def extract_hmm_parameters(model):

    # Get number of states
    n_states = len(model.distributions)

    # Extract initial probabilities (starts)
    # model.starts is a torch tensor
    Pi = model.starts.detach().numpy().astype(np.float64)

    # Extract transition matrix (edges)
    # model.edges is a torch tensor of shape (n_states, n_states)
    A = model.edges.detach().numpy().astype(np.float64)

    # Extract emission probabilities
    # Each distribution is a Categorical distribution with probs attribute
    # Need to build a matrix where B[state, observation] = P(obs | state)

    # Get the number of possible observations from the first distribution
    first_dist_probs = model.distributions[0].probs.detach().numpy()
    n_observations = first_dist_probs.shape[1]  # Should be 5 for CNA values 0-4

    # Build emission matrix
    B = np.zeros((n_states, n_observations), dtype=np.float64)
    for state_idx in range(n_states):
        dist_probs = model.distributions[state_idx].probs.detach().numpy()
        B[state_idx, :] = dist_probs.flatten()

    # Convert from log probabilities to regular probabilities if needed
    # Pomegranate may store log probabilities internally
    if np.any(A < 0):
        A = np.exp(A)
    if np.any(Pi < 0):
        Pi = np.exp(Pi)
    if np.any(B < 0):
        B = np.exp(B)

    # Ensure all values are valid probabilities (0 to 1)
    A = np.clip(A, 0, 1)
    B = np.clip(B, 0, 1)
    Pi = np.clip(Pi, 0, 1)

    # Normalize to ensure they sum to 1
    A = A / A.sum(axis=1, keepdims=True)
    B = B / B.sum(axis=1, keepdims=True)
    Pi = Pi / Pi.sum()

    return A, B, Pi

"""
    Find all genes within a genomic window using NCBI E-utilities.
    500,000 bp window surrounding IGHV3-20

    Parameters:
    -----------
    chrom : str
        Chromosome name (e.g., '14')
    start : int
        Start position
    end : int
        End position
    window_size : int
        Window size in base pairs (default: 500kb)

    Returns:
    --------
    pd.DataFrame
        DataFrame with gene information including Entrez IDs and coordinates
"""
def get_genes_in_genomic_window(chrom, start, end, window_size=500_000):
    region_start = start - window_size
    region_end = end + window_size

    print(f"Querying genes in window chr{chrom}:{region_start:,}-{region_end:,} using NCBI E-utilities...")

    # Remove 'chr' prefix if present
    chrom_num = chrom.replace('chr', '')

    # Build NCBI search query
    # Format: chromosome[CHR] AND start:end[Base Position] AND human[Organism]
    query = f"{chrom_num}[Chromosome] AND {region_start}:{region_end}[Base Position] AND human[Organism]"

    print(f"  Query: {query}")

    # Search for genes
    try:
        handle = Entrez.esearch(db="gene", term=query, retmax=1000)
        record = Entrez.read(handle)
        handle.close()

        gene_ids = record["IdList"]
        print(f"  Found {len(gene_ids)} gene IDs")

        if not gene_ids:
            print("  WARNING: No genes found in this region!")
            return pd.DataFrame(columns=['Entrez_Gene_Id', 'Chr', 'Start'])

        # Fetch gene information
        print("  Fetching gene details...")
        handle = Entrez.efetch(db="gene", id=",".join(gene_ids), retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        genes = []
        for rec in records:
            try:
                # Extract Entrez Gene ID
                track_info = rec.get('Entrezgene_track-info', {})
                gene_track = track_info.get('Gene-track', {})
                entrez_id_str = gene_track.get('Gene-track_geneid', '0')
                entrez_id = int(entrez_id_str)

                # Extract gene symbol
                gene_info = rec.get('Entrezgene_gene', {})
                symbol = gene_info.get('Gene-ref_locus', 'Unknown')

                # Extract genomic coordinates from the PRIMARY assembly (GRCh38)
                # Multiple loci may exist for different assemblies, we want the primary reference
                loci = rec.get('Entrezgene_locus', [])

                gene_start = None
                gene_end = None

                for locus in loci:
                    # Look for the primary GRCh38 assembly
                    heading = locus.get('Gene-commentary_heading', '')

                    # Prefer "Reference GRCh38" primary assembly
                    if 'Reference GRCh38' in heading and 'Primary Assembly' in heading:
                        genomic_info = locus.get('Gene-commentary_seqs', [])

                        if genomic_info:
                            for interval in genomic_info:
                                if 'Seq-loc_int' in interval:
                                    seq_int = interval['Seq-loc_int']['Seq-interval']
                                    gene_start = int(seq_int.get('Seq-interval_from', 0))
                                    gene_end = int(seq_int.get('Seq-interval_to', 0))
                                    break
                        break

                # If coordinates were found, add the gene
                if gene_start is not None and gene_end is not None:
                    genes.append({
                        'Entrez_Gene_Id': entrez_id,
                        'Symbol': symbol,
                        'Chr': chrom,
                        'Start': gene_start,
                        'End': gene_end
                    })

            except Exception as e:
                # Skip genes with parsing errors
                continue

        # Create DataFrame
        genes_df = pd.DataFrame(genes)

        if genes_df.empty:
            print("Could not parse any gene information!")
            return pd.DataFrame(columns=['Entrez_Gene_Id', 'Chr', 'Start'])

        # Sort by start position
        genes_df = genes_df.sort_values('Start').reset_index(drop=True)

        # Display sample of genes
        if len(genes_df) > 0:
            print(f"\n  Sample genes:")
            for _, row in genes_df.head(5).iterrows():
                print(f"    {row['Symbol']} (ID: {row['Entrez_Gene_Id']}) at {row['Start']:,}")

        return genes_df

    except Exception as e:
        print(f"  ERROR querying NCBI: {e}")
        print("  Returning empty DataFrame")
        return pd.DataFrame(columns=['Entrez_Gene_Id', 'Chr', 'Start'])

"""
    Load CNA data and filter for specific genes after querying the
    NCBI E-Utilities API for neighboring genes (ensures they exist
    in the current dataset)

    Parameters:
    -----------
    cna_file : CNA csv file
    entrez_ids : List of Entrez Gene IDs to filter

    Returns:
    --------
    pd.DataFrame
        Filtered CNA data
"""
def load_and_filter_cna_data(cna_file, entrez_ids):
    try:
        cna = pd.read_csv(cna_file, index_col=0)
        cna.index = cna.index.astype(int)

        # Filter for genes in the region
        cna_filtered = cna.loc[cna.index.isin(entrez_ids)]

        print(f"Loaded CNA data: {cna_filtered.shape[0]} genes, {cna_filtered.shape[1]} samples")

        return cna_filtered

    except FileNotFoundError:
        print(f"Error: CNA file not found at {cna_file}")
        raise

"""
    Prepare CNA sequences sorted by genomic position for HMM analysis.

    Parameters:
    -----------
    cna_df : pd.DataFrame, CNA data with Entrez_Gene_Id as index
    gene_positions_df : pd.DataFrame, Gene positions with Entrez_Gene_Id, Chr, Start columns

    Returns:
    --------
    tuple
        (sorted_df, sequences_dict) where sequences_dict maps patient IDs to CNA sequences
        CNA values are remapped to 0-4 range for HMM compatibility
"""
def prepare_sequences_for_hmm(cna_df, gene_positions_df):
    # Reset index to make Entrez_Gene_Id a column
    cna_df_reset = cna_df.reset_index()

    # Rename the index column if needed
    if 'index' in cna_df_reset.columns and 'Entrez_Gene_Id' not in cna_df_reset.columns:
        cna_df_reset = cna_df_reset.rename(columns={'index': 'Entrez_Gene_Id'})

    # Merge with gene positions
    df_sorted = cna_df_reset.merge(
        gene_positions_df,
        on='Entrez_Gene_Id',
        how='inner'
    )

    # Remove rows with missing coordinates
    df_sorted = df_sorted.dropna(subset=['Chr', 'Start'])

    # Sort by chromosome and start position
    df_sorted = df_sorted.sort_values(['Chr', 'Start'])

    # Extract patient sequences and remap CNA values to 0-4 range
    # Original: -2, -1, 0, 1, 2 -> Remapped: 0, 1, 2, 3, 4
    # Remapping is needed because HMM emission/transition matrices use non-negative
    # integer indices (0 to n_states-1), so negative CNA values must be shifted
    # Handle NA values by filling with 0 (neutral state = 2 after remapping)
    patient_cols = [col for col in df_sorted.columns if col.startswith("TCGA-")]
    sequences = {}

    for p in patient_cols:
        # Fill NA values with 0 (neutral) before remapping
        patient_data = df_sorted[p].fillna(0)
        # Add 2 to shift range from [-2,2] to [0,4]
        sequences[p] = (patient_data + 2).astype(int).tolist()

    print(f"Prepared {len(sequences)} patient sequences with {len(df_sorted)} genes each")

    return df_sorted, sequences

"""
    Train 2-state HMM model for CNA state prediction (neutral vs gain).

    Parameters:
    -----------
    sequences : dict
        Dictionary mapping patient IDs to CNA sequences

    Returns:
    --------
    DenseHMM
        Trained HMM model
"""
def train_hmm_model(sequences):
    # Define emission distributions for 2-state model (neutral and gain only)
    # CNA states (remapped): 0 (deep loss), 1 (loss), 2 (neutral), 3 (gain), 4 (amplification)
    # Original values: -2, -1, 0, 1, 2
    # Remapped because the Pomegranate categorical distributions expect positive integer indices

    # 2-STATE MODEL: Based on learned parameters showing loss state is essentially unused
    # State 0 = Neutral (emits -2, -1, 0 with high probability for 0)
    # State 1 = Gain (emits 1, 2 with high probability)

    # set epsilon to a small value to avoid calculation errors down the line
    epsilon = 1e-10

    # Neutral state: primarily emits 0 (neutral), but allows some deletions
    neutral_probs = np.array([0.05, 0.10, 0.75, 0.08, 0.02], dtype=np.float32) + epsilon

    # Gain state: primarily emits 1 and 2 (gain and amplification)
    gain_probs = np.array([0.0, 0.0, 0.05, 0.70, 0.25], dtype=np.float32) + epsilon

    # Normalize values so that they sum to 1
    neutral_probs = (neutral_probs / neutral_probs.sum()).astype(np.float32)
    gain_probs = (gain_probs / gain_probs.sum()).astype(np.float32)

    # Transform raw probability arrays into categorical distributions that Pomegranate accepts
    # 1) reshape into a 2D array ex: [0.05, 0.10, 0.75, 0.08, 0.02] -> [[0.05, 0.10, 0.75, 0.08, 0.02]]
    # 2) convert data points into 32-pt floating points
    # 3) wrap it inside a pomegranate categorical distribution object
    neutral_dist = Categorical(probs=neutral_probs.reshape(1, -1).astype(np.float32))
    gain_dist = Categorical(probs=gain_probs.reshape(1, -1).astype(np.float32))

    # Define transition matrix for 2-state model
    # Based on biological expectation that states are stable (high self-transition)
    transitions = np.array([
        [0.99, 0.01],  # From neutral state: stay neutral (99%), switch to gain (1%)
        [0.01, 0.99]   # From gain state: switch to neutral (1%), stay gain (99%)
    ], dtype=np.float32)

    # Initialize 2-state HMM
    model = DenseHMM(
        [neutral_dist, gain_dist],
        # Initial probabilities: favor neutral state (55%) over gain (45%)
        # Based on previous analysis showing ~45% gain frequency
        starts=np.array([0.55, 0.45], dtype=np.float32),
        edges=transitions
    )

    # Train on patient sequence of states: 0 (deep loss), 1 (loss), 2 (neutral), 3 (gain), 4 (amplification)
    # Uses Baum-Welch algorithm under the hood
    print(f"Training 2-state HMM on {len(sequences)} patient sequences...")
    seq_arrays = [np.array(seq).reshape(-1, 1) for seq in sequences.values()]
    model.fit(seq_arrays)
    print("Training complete\n")

    # Display learned parameters after training
    print("\nLearned HMM parameters (after Baum-Welch training):")

    # Extract and convert parameters
    A = model.edges.detach().numpy()
    Pi = model.starts.detach().numpy()

    if np.any(A < 0):
        A = np.exp(A)
    if np.any(Pi < 0):
        Pi = np.exp(Pi)

    A = A / A.sum(axis=1, keepdims=True)
    Pi = Pi / Pi.sum()

    # Extract emission matrix (2 states × 5 observations)
    B = np.zeros((2, 5))
    for i in range(2):
        B_row = model.distributions[i].probs.detach().numpy().flatten()
        if np.any(B_row < 0):
            B_row = np.exp(B_row)
        B[i] = B_row / B_row.sum()

    # Set print options for cleaner output
    np.set_printoptions(precision=5, suppress=True, floatmode='fixed')

    # Print transition and emission matrices and initial probabilities
    # post training
    print("\nTransition Matrix A (2×2):")
    print("         Neutral    Gain")
    print(f"Neutral  {A[0,0]}  {A[0,1]}")
    print(f"Gain     {A[1,0]}  {A[1,1]}")

    print("\nEmission Matrix B (2×5):")
    print("        -2(0)    -1(1)     0(2)     1(3)     2(4)")
    print(f"Neutral {B[0,0]}  {B[0,1]}  {B[0,2]}  {B[0,3]}  {B[0,4]}")
    print(f"Gain    {B[1,0]}  {B[1,1]}  {B[1,2]}  {B[1,3]}  {B[1,4]}")

    print("\nInitial Probabilities Pi:")
    print(f"Neutral: {Pi[0]:.5f}")
    print(f"Gain:    {Pi[1]:.5f}")
    print()

    return model

"""
    Predict hidden states for a specific patient using viterbi.

    Input Parameters:
    -----------
    model : Trained HMM model
    sequences : Dictonary of patient sequences
    patient_id : str, Patient sample ID to predict states for
"""
def predict_states(model, sequences, patient_id):
    # Get observation sequence for this patient
    obs = np.array(sequences[patient_id])

    # Extract HMM parameters from the trained model
    A, B, Pi = extract_hmm_parameters(model)

    # Run custom Viterbi algorithm
    path, T1, T2 = viterbi(obs, A, B, Pi)

    # Map state indices to names for 2-state model
    # State 0 = neutral, State 1 = gain
    state_names = ['neutral', 'gain']
    decoded_states = [state_names[int(state_idx)] for state_idx in path]

    return decoded_states

"""
    Analyze and print genomic layout and CNA frequency patterns.

    Input Parameters:
    -----------
    df_sorted : pd.DataFrame
        Sorted dataframe with genomic positions
    patient_ids : list
        List of patient IDs
"""
def analyze_regional_patterns(df_sorted, patient_ids):
    print("\n=== Genomic Layout (ordered by position) ===")
    print(df_sorted[['Entrez_Gene_Id', 'Chr', 'Start']].to_string(index=False))

    # Calculate CNA frequencies per gene
    print("\n=== CNA Frequency per Gene ===")

    deletion_freq = []

    for _, row in df_sorted.iterrows():
        gene_id = row['Entrez_Gene_Id']
        position = row['Start']

        # Count CNA states across all patients for this gene (RAW DATA)
        deletions = 0
        neutrals = 0
        gains = 0

        for patient_id in patient_ids:
            cna_value = row[patient_id]
            if cna_value < 0:
                deletions += 1
            elif cna_value == 0:
                neutrals += 1
            else:
                gains += 1

        # Count HMM-smoothed states (from Predicted_State columns)
        hmm_neutrals = 0
        hmm_gains = 0

        # Count how many patients have a neutral vs gain 
        for patient_id in patient_ids:
            pred_state_col = f'Predicted_State_{patient_id}'
            if pred_state_col in row:
                pred_state = row[pred_state_col]
                if pred_state == 'gain':
                    hmm_gains += 1
                else:  # 'neutral'
                    hmm_neutrals += 1

        total = len(patient_ids)
        deletion_freq.append({
            'Entrez_Gene_Id': gene_id,
            'Position': position,
            'Deletions': deletions,
            'Deletion_Pct': f"{(deletions/total*100):.1f}%",
            'Neutral': neutrals,
            'Neutral_Pct': f"{(neutrals/total*100):.1f}%",
            'Gains': gains,
            'Gain_Pct': f"{(gains/total*100):.1f}%",
            'HMM_Gains': hmm_gains,
            'HMM_Gain_Pct': f"{(hmm_gains/total*100):.1f}%",
            'Gain_Diff': f"{((hmm_gains-gains)/total*100):+.1f}%"
        })

    freq_df = pd.DataFrame(deletion_freq)
    print(freq_df.to_string(index=False))

    # Identify genes with highest deletion and gain frequencies
    freq_df['Del_Pct_Numeric'] = freq_df['Deletions'] / len(patient_ids) * 100
    freq_df['Gain_Pct_Numeric'] = freq_df['Gains'] / len(patient_ids) * 100
    freq_df['HMM_Gain_Pct_Numeric'] = freq_df['HMM_Gains'] / len(patient_ids) * 100

    most_deleted = freq_df.nlargest(3, 'Del_Pct_Numeric')
    most_gained = freq_df.nlargest(3, 'Gain_Pct_Numeric')
    most_gained_hmm = freq_df.nlargest(3, 'HMM_Gain_Pct_Numeric')

    print("\n=== Top Genes by Deletion Frequency (Raw Data) ===")
    print(most_deleted[['Entrez_Gene_Id', 'Position', 'Deletions', 'Deletion_Pct']].to_string(index=False))

    print("\n=== Top Genes by Gain/Amplification Frequency (Raw Data) ===")
    print(most_gained[['Entrez_Gene_Id', 'Position', 'Gains', 'Gain_Pct']].to_string(index=False))

    print("\n=== Top Genes by Gain Frequency (HMM-Smoothed) ===")
    print(most_gained_hmm[['Entrez_Gene_Id', 'Position', 'HMM_Gains', 'HMM_Gain_Pct', 'Gain_Diff']].to_string(index=False))

    # Check for regional deletion patterns (consecutive genes with high deletion rates)
    print("\n=== Regional Deletion Analysis ===")
    high_del_threshold = 40  # 40% deletion rate
    high_del_genes = freq_df[freq_df['Del_Pct_Numeric'] >= high_del_threshold]

    if len(high_del_genes) > 1:
        print(f"Genes with ≥{high_del_threshold}% deletion rate:")
        print(high_del_genes[['Entrez_Gene_Id', 'Position', 'Deletion_Pct']].to_string(index=False))
        print(f"\nThis suggests a regional deletion spanning {len(high_del_genes)} consecutive genes")
        print(f"from position {high_del_genes.iloc[0]['Position']:,} to {high_del_genes.iloc[-1]['Position']:,}")
        print(f"(~{(high_del_genes.iloc[-1]['Position'] - high_del_genes.iloc[0]['Position']):,} bp region)")
    else:
        print(f"No clear regional deletion pattern detected (threshold: {high_del_threshold}%)")

    # Check for regional gain/amplification patterns
    print("\n=== Regional Gain/Amplification Analysis (Raw Data) ===")
    high_gain_threshold = 40  # 40% gain rate
    high_gain_genes = freq_df[freq_df['Gain_Pct_Numeric'] >= high_gain_threshold]

    if len(high_gain_genes) > 1:
        print(f"Genes with >={high_gain_threshold}% gain/amplification rate (raw data):")
        print(high_gain_genes[['Entrez_Gene_Id', 'Position', 'Gain_Pct']].to_string(index=False))
        print(f"\nThis suggests a regional amplification spanning {len(high_gain_genes)} consecutive genes")
        print(f"from position {high_gain_genes.iloc[0]['Position']:,} to {high_gain_genes.iloc[-1]['Position']:,}")
        print(f"(~{(high_gain_genes.iloc[-1]['Position'] - high_gain_genes.iloc[0]['Position']):,} bp region)")
    else:
        print(f"No clear regional gain pattern detected (threshold: {high_gain_threshold}%)")

    print("\n=== Regional Gain/Amplification Analysis (HMM-Smoothed) ===")
    high_gain_genes_hmm = freq_df[freq_df['HMM_Gain_Pct_Numeric'] >= high_gain_threshold]

    if len(high_gain_genes_hmm) > 1:
        print(f"Genes with ≥{high_gain_threshold}% gain rate (HMM-smoothed):")
        print(high_gain_genes_hmm[['Entrez_Gene_Id', 'Position', 'HMM_Gain_Pct', 'Gain_Diff']].to_string(index=False))
        print(f"\nThis suggests a regional amplification spanning {len(high_gain_genes_hmm)} consecutive genes")
        print(f"from position {high_gain_genes_hmm.iloc[0]['Position']:,} to {high_gain_genes_hmm.iloc[-1]['Position']:,}")
        print(f"(~{(high_gain_genes_hmm.iloc[-1]['Position'] - high_gain_genes_hmm.iloc[0]['Position']):,} bp region)")
    else:
        print(f"No clear regional gain pattern detected after HMM smoothing (threshold: {high_gain_threshold}%)")

"""
    Export before and after smoothing CNA data to csv file
    Input Parameters:
    -----------
    df_sorted : pd.DataFrame, Sorted dataframe with genomic positions and predictions
    patient_ids : List of patient sample IDs
"""
def export_before_after_csv(df_sorted, patient_ids, output_file="hmm_before_after.csv"):
    print(f"\nExporting before/after sequences to {output_file}...")

    # State mapping for 2-state HMM predictions
    # Neutral = 0, Gain = 1
    state_map = {"neutral": 0, "gain": 1}

    # Create output dataframe with gene information
    output_df = df_sorted[['Entrez_Gene_Id', 'Chr', 'Start']].copy()

    # Collect all BEFORE and AFTER columns first
    before_after_data = {}

    for patient_id in patient_ids:
        # Raw CNA values (BEFORE) - fill NA with 0 (neutral)
        before_after_data[f'{patient_id}_BEFORE'] = df_sorted[patient_id].fillna(0)

        # HMM smoothed values (AFTER)
        pred_col = f'Predicted_State_{patient_id}'
        before_after_data[f'{patient_id}_AFTER'] = [state_map[s] for s in df_sorted[pred_col]]

    # Create DataFrame with all patient columns and concat at once
    patient_data_df = pd.DataFrame(before_after_data, index=df_sorted.index)
    output_df = pd.concat([output_df, patient_data_df], axis=1)

    # Save to CSV
    output_df.to_csv(output_file, index=False)

    print(f"Successfully exported {len(output_df)} genes × {len(patient_ids)} patients")
    print(f"Columns: Entrez_Gene_Id, Chr, Start, <PatientID>_BEFORE, <PatientID>_AFTER")
    print(f"States: neutral=0, gain=1")
    print(f"File saved: {output_file}\n")

    return output_df

"""
    Create heatmaps showing raw CNA data vs HMM-smoothed data for all patients,
    with a diff heatmap

    Input Parameters:
    -----------
    df_sorted : Sorted dataframe (by genomic regions) with genomic positions and predictions
    patient_ids : List of patient IDs
"""
def plot_before_after_heatmaps(df_sorted, patient_ids):
    # State mapping for 2-state HMM predictions
    # Neutral = 0, Gain = 1
    state_map = {"neutral": 0, "gain": 1}

    # Create matrices for raw and smoothed data
    n_genes = len(df_sorted)
    n_patients = len(patient_ids)

    raw_matrix = np.zeros((n_genes, n_patients))
    smoothed_matrix = np.zeros((n_genes, n_patients))

    for i, patient_id in enumerate(patient_ids):
        # fill in raw CNA values column by column for each sample
        raw_matrix[:, i] = df_sorted[patient_id].values

        # HMM smoothed values
        pred_col = f'Predicted_State_{patient_id}'
        smoothed_matrix[:, i] = [state_map[s] for s in df_sorted[pred_col]]

    # Calculate difference matrix
    diff_matrix = raw_matrix - smoothed_matrix

    # Calculate quantitative metrics
    total_transitions_raw = 0
    total_transitions_smoothed = 0

    for i in range(n_patients):
        # Count state transitions in raw data
        for j in range(n_genes - 1):
            if raw_matrix[j, i] != raw_matrix[j + 1, i]:
                total_transitions_raw += 1

        # Count state transitions in smoothed data
        for j in range(n_genes - 1):
            if smoothed_matrix[j, i] != smoothed_matrix[j + 1, i]:
                total_transitions_smoothed += 1

    # Calculate percentage of calls that changed
    total_calls = n_genes * n_patients
    calls_changed = np.sum(diff_matrix != 0)
    pct_changed = (calls_changed / total_calls) * 100

    # Print metrics
    print("\n=== Noise Reduction Metrics ===")
    print(f"Total CNA calls: {total_calls:,}")
    print(f"Calls changed by HMM: {calls_changed} ({pct_changed:.1f}%)")
    print(f"Total transitions BEFORE: {total_transitions_raw}")
    print(f"Total transitions AFTER: {total_transitions_smoothed}")
    print(f"Transitions reduced by: {total_transitions_raw - total_transitions_smoothed} ({((total_transitions_raw - total_transitions_smoothed) / total_transitions_raw * 100):.1f}%)")

    # Scale figure height based on number of genes (min 8, max 12 for better display)
    fig_height = min(max(8, n_genes / 20), 12)
    fig_width = 10

    # Determine y-axis labeling strategy - always include IGHV genes
    ighv_gene_ids = [28445, 28444, 28443, 28374]  # IGHV3-20, IGHV3-21, IGHV3-22, IGHVII-22-1

    # Find the row position of IGHV3-20 (28445) for horizontal line
    ighv3_20_position = None
    if 28445 in df_sorted['Entrez_Gene_Id'].values:
        idx = df_sorted[df_sorted['Entrez_Gene_Id'] == 28445].index[0]
        ighv3_20_position = df_sorted.index.get_loc(idx)

    if n_genes <= 30:
        # Show all gene labels
        ytick_positions = range(n_genes)
        ytick_labels = df_sorted['Entrez_Gene_Id'].values
        label_fontsize = 6
    else:
        # Show sparse labels with position info, ALWAYS including IGHV genes
        # Reduce density to avoid overcrowding
        step = max(n_genes // 10, 1)  # Fewer labels overall
        ytick_positions = list(range(0, n_genes, step))

        # Find positions of IGHV genes and ensure they're included
        ighv_positions = []
        for gene_id in ighv_gene_ids:
            if gene_id in df_sorted['Entrez_Gene_Id'].values:
                idx = df_sorted[df_sorted['Entrez_Gene_Id'] == gene_id].index[0]
                gene_row_position = df_sorted.index.get_loc(idx)
                ighv_positions.append(gene_row_position)

                # Only add if not too close to existing labels (at least 5 rows apart)
                if not any(abs(gene_row_position - pos) < 5 for pos in ytick_positions):
                    ytick_positions.append(gene_row_position)

        # Sort positions for proper display
        ytick_positions.sort()

        # Create labels - show both ID and position for all genes
        ytick_labels = []
        for i in ytick_positions:
            gene_id = df_sorted.iloc[i]['Entrez_Gene_Id']
            position = df_sorted.iloc[i]['Start']
            # Show both ID and position for all genes
            ytick_labels.append(f"{gene_id}\n{position:,}")

        label_fontsize = 7

    # FIGURE 1: Raw CNA Data
    print("\nGenerating Figure 1: Raw CNA Data...")
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    im1 = ax1.imshow(raw_matrix, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax1.set_title(f'BEFORE: Raw CNA Data\n{n_patients} patients, {n_genes} genes in IGHV3-20 region (±500kb)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Patients', fontsize=11)
    ax1.set_ylabel('Genes (sorted by genomic position)', fontsize=11)
    ax1.set_yticks(ytick_positions)
    ax1.set_yticklabels(ytick_labels, fontsize=label_fontsize)

    # Color-code IGHV gene labels in red for visibility
    for idx, (tick_pos, label) in enumerate(zip(ytick_positions, ytick_labels)):
        gene_id = df_sorted.iloc[tick_pos]['Entrez_Gene_Id']
        if gene_id in ighv_gene_ids:
            ax1.get_yticklabels()[idx].set_color('red')
            ax1.get_yticklabels()[idx].set_weight('bold')

    # Add horizontal line at IGHV3-20 position
    if ighv3_20_position is not None:
        ax1.axhline(y=ighv3_20_position, color='red', linestyle='--', linewidth=2, alpha=0.7)

    plt.colorbar(im1, ax=ax1, label='CNA State', fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('heatmap_1_raw_cna.png', dpi=150, bbox_inches='tight')
    print("Saved: heatmap_1_raw_cna.png")
    plt.close(fig1)

    # FIGURE 2: HMM Smoothed Data
    print("Generating Figure 2: HMM Smoothed Data...")
    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
    im2 = ax2.imshow(smoothed_matrix, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax2.set_title(f'AFTER: HMM-Smoothed (Viterbi) Data\n{n_patients} patients, {n_genes} genes in IGHV3-20 region (±500kb)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Patients', fontsize=11)
    ax2.set_ylabel('Genes (sorted by genomic position)', fontsize=11)
    ax2.set_yticks(ytick_positions)
    ax2.set_yticklabels(ytick_labels, fontsize=label_fontsize)

    # Color-code IGHV gene labels in red for visibility
    for idx, (tick_pos, label) in enumerate(zip(ytick_positions, ytick_labels)):
        gene_id = df_sorted.iloc[tick_pos]['Entrez_Gene_Id']
        if gene_id in ighv_gene_ids:
            ax2.get_yticklabels()[idx].set_color('red')
            ax2.get_yticklabels()[idx].set_weight('bold')

    # Add horizontal line at IGHV3-20 position
    if ighv3_20_position is not None:
        ax2.axhline(y=ighv3_20_position, color='red', linestyle='--', linewidth=2, alpha=0.7)

    plt.colorbar(im2, ax=ax2, label='CNA State', fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('heatmap_2_hmm_smoothed.png', dpi=150, bbox_inches='tight')
    print("Saved: heatmap_2_hmm_smoothed.png")
    plt.close(fig2)

    # FIGURE 3: Difference Heatmap
    print("Generating Figure 3: Difference Heatmap...")
    fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))
    im3 = ax3.imshow(diff_matrix, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax3.set_title(f'DIFFERENCE: Raw - Smoothed\n{pct_changed:.1f}% calls changed | Transitions: {total_transitions_raw} → {total_transitions_smoothed} ({((total_transitions_raw - total_transitions_smoothed) / total_transitions_raw * 100):.1f}% reduction)',
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Patients', fontsize=11)
    ax3.set_ylabel('Genes (sorted by genomic position)', fontsize=11)
    ax3.set_yticks(ytick_positions)
    ax3.set_yticklabels(ytick_labels, fontsize=label_fontsize)

    # Color-code IGHV gene labels in red for visibility
    for idx, (tick_pos, label) in enumerate(zip(ytick_positions, ytick_labels)):
        gene_id = df_sorted.iloc[tick_pos]['Entrez_Gene_Id']
        if gene_id in ighv_gene_ids:
            ax3.get_yticklabels()[idx].set_color('red')
            ax3.get_yticklabels()[idx].set_weight('bold')

    # Add horizontal line at IGHV3-20 position
    if ighv3_20_position is not None:
        ax3.axhline(y=ighv3_20_position, color='red', linestyle='--', linewidth=2, alpha=0.7)

    cbar3 = plt.colorbar(im3, ax=ax3, label='Difference (Raw - Smoothed)', fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('heatmap_3_difference.png', dpi=150, bbox_inches='tight')
    print("Saved: heatmap_3_difference.png")
    plt.close(fig3)

def main():
    # Configuration - IGHV3-20 (Entrez ID: 28445) on chromosome 14
    CNA_FILE = "cna_data.csv"
    TARGET_GENE = "IGHV3-20"
    TARGET_ENTREZ_ID = 28445
    TARGET_CHROM = '14'
    TARGET_START = 106210936  # IGHV3-20 start position
    TARGET_END = 106211453    # IGHV3-20 end position
    WINDOW_SIZE = 500_000     # ±500kb window around IGHV3-20

    print(f"=== IGHV3-20 Region Analysis ===")
    print(f"Target gene: {TARGET_GENE} (Entrez ID: {TARGET_ENTREZ_ID})")
    print(f"Location: chr{TARGET_CHROM}:{TARGET_START}-{TARGET_END}")
    print(f"Analysis window: ±{WINDOW_SIZE:,}bp\n")

    # Step 1: Get genes in genomic window around IGHV3-20 using NCBI E-utilities
    print("Step 1: Finding genes in genomic window around IGHV3-20...")
    genes_df = get_genes_in_genomic_window(
        TARGET_CHROM, TARGET_START, TARGET_END, WINDOW_SIZE
    )

    # Verify IGHV genes are in the results
    # NCBI E-utilities should return immunoglobulin genes, these are hardcoded 
    # after determining neighboring gene annotations in Ensembl
    ighv_genes = [
        {'entrez_id': 28445, 'name': 'IGHV3-20'},
        {'entrez_id': 28444, 'name': 'IGHV3-21'},
        {'entrez_id': 28443, 'name': 'IGHV3-22'},
        {'entrez_id': 28374, 'name': 'IGHVII-22-1'}
    ]

    print("\nVerifying IGHV genes in retrieved results:")
    for gene in ighv_genes:
        if gene['entrez_id'] in genes_df['Entrez_Gene_Id'].values:
            gene_info = genes_df[genes_df['Entrez_Gene_Id'] == gene['entrez_id']].iloc[0]
            print(f"{gene['name']} (Entrez ID: {gene['entrez_id']}) found at position {gene_info['Start']:,}")
        else:
            print(f"{gene['name']} (Entrez ID: {gene['entrez_id']}) NOT found")

    print(f"\nTotal genes in analysis: {len(genes_df)}")

    # Step 2: Load and filter CNA data
    print("\n Loading and filtering CNA data...")
    cna_data = load_and_filter_cna_data(CNA_FILE, genes_df['Entrez_Gene_Id'].tolist())

    # Check if IGHV genes are in the CNA data
    print("\nChecking IGHV genes in CNA data:")
    patient_cols = [col for col in cna_data.columns if col.startswith("TCGA-")]
    for gene in ighv_genes:
        if gene['entrez_id'] in cna_data.index:
            ighv_cna = cna_data.loc[gene['entrez_id']]
            ighv_patient_data = ighv_cna[patient_cols]
            non_zero = (ighv_patient_data.abs() > 0).sum()
            print(f" {gene['name']} (Entrez ID: {gene['entrez_id']}): {non_zero} non-zero CNA calls out of {len(patient_cols)} patients")
        else:
            print(f" WARNING: {gene['name']} (Entrez ID: {gene['entrez_id']}) NOT found in CNA data!")



    print("\n Preparing sequences for HMM...")
    df_sorted, sequences = prepare_sequences_for_hmm(cna_data, genes_df)


    print("\n Training HMM model...")
    model = train_hmm_model(sequences)

    all_predictions = {}
    patient_ids = list(sequences.keys())

    for i, patient_id in enumerate(patient_ids):
        predicted_states = predict_states(model, sequences, patient_id)
        all_predictions[patient_id] = predicted_states

    # Add all prediction columns at once to avoid fragmentation
    print(f"Completed predictions for all {len(patient_ids)} patients")
    print("Adding predictions to dataframe...")

  
    # Create predictions dataframe with explicit index matching
    predictions_df = pd.DataFrame(
        {f'Predicted_State_{pid}': all_predictions[pid] for pid in patient_ids},
        index=df_sorted.index  # Explicitly set index to match df_sorted
    )
    
    df_sorted = pd.concat([df_sorted, predictions_df], axis=1)

    # Re-sort by genomic position to ensure correct ordering for visualization
    print("Re-sorting genes by genomic position...")
    df_sorted = df_sorted.sort_values('Start').reset_index(drop=True)
    print(f"Genes sorted by position. Order: {df_sorted['Entrez_Gene_Id'].tolist()}")

    analyze_regional_patterns(df_sorted, patient_ids)

    # Export before/after data to CSV
    export_before_after_csv(df_sorted, patient_ids)

    plot_before_after_heatmaps(df_sorted, patient_ids)

    return df_sorted, model, sequences, all_predictions


if __name__ == "__main__":
    df_sorted, model, sequences, all_predictions = main()
