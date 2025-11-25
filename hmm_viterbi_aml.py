#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMM-based Copy Number Alteration Analysis for IGHV3-20 Region

This script performs genomic region analysis and HMM-based CNA state prediction
on TCGA patient data, focusing on the IGHV3-20 gene region on chromosome 14.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from biomart import BiomartServer
from io import StringIO


def viterbi(y, A, B, Pi=None):
    """
    Viterbi Algorithm Implementation using log probabilities to avoid numerical underflow.

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
    # gets number of hidden states (3 in this case for loss, neutral, gain)
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
    # rows (K) - each hidden state, 0 loss 1 neutral 2 gain
    # cols (T) - each gene 
    # T1 holds log probability of being at a particular state (K) at a particular gene (T)
    # ex T1:
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


def extract_hmm_parameters(model):
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
    import torch

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
    # We need to build a matrix where B[state, observation] = P(obs | state)

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


def get_genes_in_genomic_window(chrom, start, end, window_size=500_000):
    """
    Find all genes within a genomic window using Biomart.

    Parameters:
    -----------
    chrom : str
        Chromosome name
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
    region_start = start - window_size
    region_end = end + window_size

    print(f"Querying genes in window chr{chrom}:{region_start}-{region_end}")

    server = BiomartServer("http://www.ensembl.org/biomart")
    dataset = server.datasets['hsapiens_gene_ensembl']

    response = dataset.search({
        'filters': {
            'chromosome_name': str(chrom),
            'start': region_start,
            'end': region_end
        },
        'attributes': ['entrezgene_id', 'chromosome_name', 'start_position']
    })

    genes_df = pd.read_csv(
        StringIO(response.text),
        sep="\t",
        header=None,
        names=['Entrez_Gene_Id', 'Chr', 'Start']
    )

    # Clean and convert data types
    genes_df['Entrez_Gene_Id'] = pd.to_numeric(
        genes_df['Entrez_Gene_Id'], errors='coerce'
    ).astype('Int64')
    genes_df['Start'] = pd.to_numeric(
        genes_df['Start'], errors='coerce'
    ).astype('Int64')
    genes_df.dropna(subset=['Entrez_Gene_Id', 'Chr', 'Start'], inplace=True)
    genes_df['Entrez_Gene_Id'] = genes_df['Entrez_Gene_Id'].astype(int)
    genes_df['Start'] = genes_df['Start'].astype(int)

    # Sort by start position
    genes_df = genes_df.sort_values('Start').reset_index(drop=True)

    print(f"Found {len(genes_df)} genes in the region")

    return genes_df


def load_and_filter_cna_data(cna_file, entrez_ids):
    """
    Load CNA data and filter for specific genes.

    Parameters:
    -----------
    cna_file : str
        Path to CNA CSV file
    entrez_ids : list
        List of Entrez Gene IDs to filter

    Returns:
    --------
    pd.DataFrame
        Filtered CNA data
    """
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


def analyze_gene_variance(cna_df):
    """
    Calculate variance in CNA states across patients.

    Parameters:
    -----------
    cna_df : pd.DataFrame
        CNA data with genes as rows and patients as columns

    Returns:
    --------
    pd.DataFrame
        Genes sorted by variance (number of non-zero CNA states)
    """
    patient_cols = [col for col in cna_df.columns if col.startswith("TCGA-")]
    cna_patient_data = cna_df[patient_cols]

    non_zero_counts = (cna_patient_data.abs() > 0).sum(axis=1)

    variance_df = pd.DataFrame({
        'Entrez_Gene_Id': non_zero_counts.index,
        'Non_Zero_CNA_Count': non_zero_counts.values
    }).sort_values(by='Non_Zero_CNA_Count', ascending=False)

    return variance_df


def prepare_sequences_for_hmm(cna_df, gene_positions_df):
    """
    Prepare CNA sequences sorted by genomic position for HMM analysis.

    Parameters:
    -----------
    cna_df : pd.DataFrame
        CNA data with Entrez_Gene_Id as index
    gene_positions_df : pd.DataFrame
        Gene positions with Entrez_Gene_Id, Chr, Start columns

    Returns:
    --------
    tuple
        (sorted_df, sequences_dict) where sequences_dict maps patient IDs to CNA sequences
        CNA values are remapped to 0-4 range for HMM compatibility
    """
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
    patient_cols = [col for col in df_sorted.columns if col.startswith("TCGA-")]
    sequences = {}
    for p in patient_cols:
        # Add 2 to shift range from [-2,2] to [0,4]
        sequences[p] = (df_sorted[p] + 2).astype(int).tolist()

    print(f"Prepared {len(sequences)} patient sequences with {len(df_sorted)} genes each")
    print("CNA values remapped: -2→0, -1→1, 0→2, 1→3, 2→4")

    return df_sorted, sequences


def train_hmm_model(sequences):
    """
    Train HMM model for CNA state prediction.

    Parameters:
    -----------
    sequences : dict
        Dictionary mapping patient IDs to CNA sequences

    Returns:
    --------
    DenseHMM
        Trained HMM model
    """
    from pomegranate.hmm import DenseHMM
    from pomegranate.distributions import Categorical

    # Define emission distributions
    # CNA states (remapped): 0 (deep loss), 1 (loss), 2 (neutral), 3 (gain), 4 (amplification)
    # Original values: -2, -1, 0, 1, 2
    # Remapped because the Progranate categorical distrubitions expect positive integer indices
        # loss_probs = [0.5, 0.5, 0.0, 0.0, 0.0]
        #           idx  0    1    2    3    4
        # if observation was a loss, using original value means loss_probs[-1] -> 0.0
        # but it should be 0.5, so the states need to be remapped to positive indices in order
        # to access the probabilities correctly

    # set epsilon to a small value to avoid calculation errors down the line
    epsilon = 1e-10
    loss_probs = np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32) + epsilon  # Emits 0 or 1
    neutral_probs = np.array([0.0, 0.1, 0.8, 0.1, 0.0], dtype=np.float32) + epsilon  # Emits mostly 2
    gain_probs = np.array([0.0, 0.0, 0.1, 0.6, 0.3], dtype=np.float32) + epsilon  # Emits 3 or 4

    # Normalize values so that it sums to 1
    loss_probs = (loss_probs / loss_probs.sum()).astype(np.float32)
    neutral_probs = (neutral_probs / neutral_probs.sum()).astype(np.float32)
    gain_probs = (gain_probs / gain_probs.sum()).astype(np.float32)

    # transform raw probability array into categorical distributions that Pomegranate accepts
    # 1) reshape into a 2D array ex: [0.5, 0.5, 0.0, 0.0, 0.0] ->[[0.5, 0.5, 0.0, 0.0, 0.0]]
    # 2) convert data points into 32-pt floating points 
    # 3) wrap it inside a pomegranate categorical distribution object 
    loss_dist = Categorical(probs=loss_probs.reshape(1, -1).astype(np.float32))
    neutral_dist = Categorical(probs=neutral_probs.reshape(1, -1).astype(np.float32))
    gain_dist = Categorical(probs=gain_probs.reshape(1, -1).astype(np.float32))

    # Define transition matrix - this is based off heuristic that staying in the 
    # same state is biologically probable
    transitions = np.array([
        [0.995, 0.005, 0.000],  # From loss state
        [0.005, 0.990, 0.005],  # From neutral state
        [0.000, 0.005, 0.995]   # From gain state
    ], dtype=np.float32)

    # Initialize HMM
    model = DenseHMM(
        [loss_dist, neutral_dist, gain_dist],
        # sets initial probability to favor staying in the same, neutral state
        starts=np.array([0.025, 0.95, 0.025], dtype=np.float32),  
        edges=transitions
    )

    # Train on patient sequence of states  0 (deep loss), 1 (loss), 2 (neutral), 3 (gain), 4 (amplification)
    # uses Baum Welch algorithm under the hood
    print(f"Training HMM on {len(sequences)} patient sequences...")
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

    # Extract emission matrix
    B = np.zeros((3, 5))
    for i in range(3):
        B_row = model.distributions[i].probs.detach().numpy().flatten()
        if np.any(B_row < 0):
            B_row = np.exp(B_row)
        B[i] = B_row / B_row.sum()

    # Set print options for cleaner output
    np.set_printoptions(precision=5, suppress=True, floatmode='fixed')

    print("\nTransition Matrix A:")
    print(A)
    print("\nEmission Matrix B:")
    print(B)
    print("\nInitial Probabilities Pi:")
    print(Pi)
    print()

    return model


def predict_states(model, sequences, patient_id):
    """
    Predict hidden states for a specific patient using viterbi.

    Parameters:
    -----------
    model : DenseHMM
        Trained HMM model
    sequences : dict
        Dictionary of patient sequences
    patient_id : str
        Patient sample ID to predict states for

    Returns:
    --------
    list
        Predicted state names (loss, neutral, or gain)
    """
    # Get observation sequence for this patient
    obs = np.array(sequences[patient_id])

    # Extract HMM parameters from the trained model
    A, B, Pi = extract_hmm_parameters(model)

    # Run custom Viterbi algorithm
    path, T1, T2 = viterbi(obs, A, B, Pi)

    # Map state indices to names
    # State 0 = loss, State 1 = neutral, State 2 = gain
    state_names = ['loss', 'neutral', 'gain']
    decoded_states = [state_names[int(state_idx)] for state_idx in path]

    return decoded_states


def analyze_regional_patterns(df_sorted, patient_ids):
    """
    Analyze and print genomic layout and CNA frequency patterns.

    Parameters:
    -----------
    df_sorted : pd.DataFrame
        Sorted dataframe with genomic positions
    patient_ids : list
        List of patient IDs
    """
    print("\n=== Genomic Layout (ordered by position) ===")
    print(df_sorted[['Entrez_Gene_Id', 'Chr', 'Start']].to_string(index=False))

    # Calculate CNA frequencies per gene
    print("\n=== CNA Frequency per Gene ===")

    deletion_freq = []
    neutral_freq = []
    gain_freq = []

    for idx, row in df_sorted.iterrows():
        gene_id = row['Entrez_Gene_Id']
        position = row['Start']

        # Count CNA states across all patients for this gene
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

        total = len(patient_ids)
        deletion_freq.append({
            'Entrez_Gene_Id': gene_id,
            'Position': position,
            'Deletions': deletions,
            'Deletion_Pct': f"{(deletions/total*100):.1f}%",
            'Neutral': neutrals,
            'Neutral_Pct': f"{(neutrals/total*100):.1f}%",
            'Gains': gains,
            'Gain_Pct': f"{(gains/total*100):.1f}%"
        })

    freq_df = pd.DataFrame(deletion_freq)
    print(freq_df.to_string(index=False))

    # Identify genes with highest deletion and gain frequencies
    freq_df['Del_Pct_Numeric'] = freq_df['Deletions'] / len(patient_ids) * 100
    freq_df['Gain_Pct_Numeric'] = freq_df['Gains'] / len(patient_ids) * 100

    most_deleted = freq_df.nlargest(3, 'Del_Pct_Numeric')
    most_gained = freq_df.nlargest(3, 'Gain_Pct_Numeric')

    print("\n=== Top Genes by Deletion Frequency ===")
    print(most_deleted[['Entrez_Gene_Id', 'Position', 'Deletions', 'Deletion_Pct']].to_string(index=False))

    print("\n=== Top Genes by Gain/Amplification Frequency ===")
    print(most_gained[['Entrez_Gene_Id', 'Position', 'Gains', 'Gain_Pct']].to_string(index=False))

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
    print("\n=== Regional Gain/Amplification Analysis ===")
    high_gain_threshold = 40  # 40% gain rate
    high_gain_genes = freq_df[freq_df['Gain_Pct_Numeric'] >= high_gain_threshold]

    if len(high_gain_genes) > 1:
        print(f"Genes with ≥{high_gain_threshold}% gain/amplification rate:")
        print(high_gain_genes[['Entrez_Gene_Id', 'Position', 'Gain_Pct']].to_string(index=False))
        print(f"\nThis suggests a regional amplification spanning {len(high_gain_genes)} consecutive genes")
        print(f"from position {high_gain_genes.iloc[0]['Position']:,} to {high_gain_genes.iloc[-1]['Position']:,}")
        print(f"(~{(high_gain_genes.iloc[-1]['Position'] - high_gain_genes.iloc[0]['Position']):,} bp region)")
    else:
        print(f"No clear regional gain pattern detected (threshold: {high_gain_threshold}%)")




def export_before_after_csv(df_sorted, patient_ids, output_file="hmm_before_after.csv"):
    """
    Export raw CNA data and HMM-smoothed predictions to CSV file.

    Parameters:
    -----------
    df_sorted : pd.DataFrame
        Sorted dataframe with genomic positions and predictions
    patient_ids : list
        List of patient IDs
    output_file : str
        Output CSV filename (default: "hmm_before_after.csv")
    """
    print(f"\nExporting before/after sequences to {output_file}...")

    # State mapping for HMM predictions
    state_map = {"loss": -1, "neutral": 0, "gain": 1}

    # Create output dataframe with gene information
    output_df = df_sorted[['Entrez_Gene_Id', 'Chr', 'Start']].copy()

    # Collect all BEFORE and AFTER columns first
    before_after_data = {}

    for patient_id in patient_ids:
        # Raw CNA values (BEFORE)
        before_after_data[f'{patient_id}_BEFORE'] = df_sorted[patient_id]

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
    print(f"File saved: {output_file}\n")

    return output_df

def plot_before_after_heatmaps(df_sorted, patient_ids):
    """
    Create side-by-side heatmaps showing raw CNA data vs HMM-smoothed data for all patients,
    with a diff heatmap 

    Parameters:
    -----------
    df_sorted : pd.DataFrame
        Sorted dataframe (by genomic regions) with genomic positions and predictions
    patient_ids : list
        List of patient IDs
    """
    
    # state mapping for HMM predictions
    # this is coming from the metadata, loss encompasses -2 and -1, neutral is 0, gain encompasses 1 and 2
    state_map = {"loss": -1, "neutral": 0, "gain": 1}

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

     # Create three-panel figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Raw CNA heatmap
    im1 = ax1.imshow(raw_matrix, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax1.set_title('BEFORE: Raw CNA Data', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Patients')
    ax1.set_ylabel('Genes (sorted by genomic position)')
    ax1.set_yticks(range(n_genes))
    ax1.set_yticklabels(df_sorted['Entrez_Gene_Id'].values, fontsize=8)
    plt.colorbar(im1, ax=ax1, label='CNA State')

    # HMM smoothed heatmap
    im2 = ax2.imshow(smoothed_matrix, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax2.set_title('AFTER: HMM-Smoothed (Viterbi) Data', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Patients')
    ax2.set_ylabel('Genes (sorted by genomic position)')
    ax2.set_yticks(range(n_genes))
    ax2.set_yticklabels(df_sorted['Entrez_Gene_Id'].values, fontsize=8)
    plt.colorbar(im2, ax=ax2, label='CNA State')

    # Difference heatmap
    im3 = ax3.imshow(diff_matrix, aspect='auto', cmap='PiYG', vmin=-2, vmax=2)
    ax3.set_title(f'DIFFERENCE: Raw - Smoothed\n({pct_changed:.1f}% calls changed)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Patients')
    ax3.set_ylabel('Genes (sorted by genomic position)')
    ax3.set_yticks(range(n_genes))
    ax3.set_yticklabels(df_sorted['Entrez_Gene_Id'].values, fontsize=8)
    plt.colorbar(im3, ax=ax3, label='Difference')

    plt.suptitle(f'Noise Reduction via HMM: {n_patients} patients, {n_genes} genes in IGHV3-20 region\n'
                 f'Transitions reduced: {total_transitions_raw} → {total_transitions_smoothed} ({((total_transitions_raw - total_transitions_smoothed) / total_transitions_raw * 100):.1f}% reduction)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
  
def main():
    """Main analysis pipeline for IGHV3-20 region analysis."""

    # Configuration - IGHV3-20 (Entrez ID: 28445) on chromosome 14
    CNA_FILE = "cna_bioinf.csv"
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

    # Step 1: Get genes in genomic window around IGHV3-20
    print("Step 1: Finding genes in genomic window around IGHV3-20...")
    genes_df = get_genes_in_genomic_window(
        TARGET_CHROM, TARGET_START, TARGET_END, WINDOW_SIZE
    )

    # Check if IGHV genes are in the results (Biomart often doesn't return immunoglobulin genes)
    # IGHV genes to check and add manually if missing
    ighv_genes = [
        {'entrez_id': 28445, 'name': 'IGHV3-20', 'start': 106210936},
        {'entrez_id': 28444, 'name': 'IGHV3-21', 'start': 106235062},
        {'entrez_id': 28443, 'name': 'IGHV3-22', 'start': 106257762},
        {'entrez_id': 28374, 'name': 'IGHVII-22-1', 'start': 106263360}
    ]

    missing_genes = []
    for gene in ighv_genes:
        if gene['entrez_id'] not in genes_df['Entrez_Gene_Id'].values:
            missing_genes.append(gene)
            print(f" {gene['name']} (Entrez ID: {gene['entrez_id']}) NOT found in Biomart results")
        else:
            print(f" {gene['name']} (Entrez ID: {gene['entrez_id']}) found in Biomart results")

    if missing_genes:
        print(f"\nAdding {len(missing_genes)} missing IGHV gene(s) manually with known coordinates...")

        # Create DataFrame with all missing genes
        new_genes = pd.DataFrame({
            'Entrez_Gene_Id': [g['entrez_id'] for g in missing_genes],
            'Chr': [TARGET_CHROM] * len(missing_genes),
            'Start': [g['start'] for g in missing_genes]
        })

        genes_df = pd.concat([genes_df, new_genes], ignore_index=True)
        genes_df = genes_df.sort_values('Start').reset_index(drop=True)

        for gene in missing_genes:
            print(f"   Added {gene['name']} (Entrez ID: {gene['entrez_id']}) at position {gene['start']:,}")
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
    print(f"  Completed predictions for all {len(patient_ids)} patients")
    print("  Adding predictions to dataframe...")

  
    # Create predictions dataframe with explicit index matching
    predictions_df = pd.DataFrame(
        {f'Predicted_State_{pid}': all_predictions[pid] for pid in patient_ids},
        index=df_sorted.index  # Explicitly set index to match df_sorted
    )
    
    df_sorted = pd.concat([df_sorted, predictions_df], axis=1)

    # Re-sort by genomic position to ensure correct ordering for visualization
    print("  Re-sorting genes by genomic position...")
    df_sorted = df_sorted.sort_values('Start').reset_index(drop=True)
    print(f"  Genes sorted by position. Order: {df_sorted['Entrez_Gene_Id'].tolist()}")

    analyze_regional_patterns(df_sorted, patient_ids)

    plot_before_after_heatmaps(df_sorted, patient_ids)

    return df_sorted, model, sequences, all_predictions


if __name__ == "__main__":
    df_sorted, model, sequences, all_predictions = main()
