import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import genextreme, t
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

output_folder = '<path_to_output>'

def trim_data(data1, data2, timeframe):
    """Trim data to the specified time range."""
    return data1[:, :, timeframe[0]:timeframe[1] + 1], data2[:, :, timeframe[0]:timeframe[1] + 1]

def load_data(data1_path, data2_path):
    """Load data and handle errors."""
    mat1_contents = loadmat(data1_path)
    mat2_contents = loadmat(data2_path)
    
    try:
        data1 = mat1_contents['data']
        data2 = mat2_contents['data']
        OTU_ID = mat1_contents['OTU_ID']
        OTU_names = mat1_contents['OTU_names']
    except KeyError as e:
        print(f"KeyError: {e}. One of the expected keys is missing in the .mat file.")
        raise

    return data1, data2, OTU_ID, OTU_names

def calculate_covariance_and_reorder(data1_trm, data2_trm):
    """Calculate covariance and reorder it."""
    N_taxa1, N_month1 = data1_trm.shape[1], data1_trm.shape[2]
    N_taxa2, N_month2 = data2_trm.shape[1], data2_trm.shape[2]

    cov_data = np.zeros((N_taxa2, N_taxa2, N_month2))
    cov_data_reordered = np.zeros((N_taxa2, N_taxa2, N_month2))
    reordered_indices = np.zeros((N_month2, N_taxa2), dtype=int)

    for i in range(N_month2):
        valid_data1 = data1_trm[:, :, i]
        valid_data2 = data2_trm[:, :, i]
        cov_data1 = np.cov(valid_data1, rowvar=False, bias=True)
        cov_data2 = np.cov(valid_data2, rowvar=False, bias=True)
        cov_data[:, :, i] = cov_data1 - cov_data2

        Z = linkage(cov_data[:, :, i])
        reordered_indices[i, :] = leaves_list(Z)
        cov_data_reordered[:, :, i] = cov_data[:, :, i][reordered_indices[i, :], :][:, reordered_indices[i, :]]
        cov_data_reordered[:, :, i] /= np.max(cov_data_reordered[:, :, i]) if np.max(cov_data_reordered[:, :, i]) != 0 else 1

    return cov_data, reordered_indices, cov_data_reordered

def select_taxa(reordered_indices, threshold, OTU_names, OTU_ID, cov_data):
    """Select important taxa."""
    monthly_relevant_taxa_covariance = [idx[-threshold:] for idx in reordered_indices]
    monthly_relevant_taxa_unique = np.unique(np.concatenate(monthly_relevant_taxa_covariance))
    
    taxa_names = OTU_names[monthly_relevant_taxa_unique]
    OTU_ID_trm = OTU_ID[monthly_relevant_taxa_unique]

    cov_data_uniq = np.array([cov_data[np.ix_(monthly_relevant_taxa_unique, monthly_relevant_taxa_unique, [i])].squeeze() 
                               for i in range(cov_data.shape[2])])
    
    pd.DataFrame(taxa_names).to_csv(os.path.join(output_folder, 'taxa_names.csv'), index=False)
    pd.DataFrame(OTU_ID_trm).to_csv(os.path.join(output_folder, 'OTU_ID_trm.csv'), index=False)

    return monthly_relevant_taxa_unique, taxa_names, OTU_ID_trm, cov_data_uniq

def normalize_cov_data(cov_data_uniq):
    """Normalize covariance data."""
    max_value = np.max(cov_data_uniq, axis=(0, 1))
    return np.zeros_like(cov_data_uniq) if np.all(max_value == 0) else cov_data_uniq / max_value

def perform_eigendecomposition(cov_data_uniq_norm, eigenvector_count):
    """Perform eigen decomposition."""
    y_fracs, ev_wtss = [], []
    
    for i in range(cov_data_uniq_norm.shape[2]):
        cov_mat = cov_data_uniq_norm[:, :, i]
        eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        
        y_frac = eigenvalues / np.sum(eigenvalues)
        ev_wts = eigenvectors[:, eigenvector_count - 1]
        
        y_fracs.append(y_frac)
        ev_wtss.append(ev_wts)
        
        np.savetxt(os.path.join(output_folder, f'y_frac_{i + 1}.csv'), y_frac)
        np.savetxt(os.path.join(output_folder, f'ev_wts_{i + 1}.csv'), ev_wts)

    return y_fracs, ev_wtss
  
def plot_eigenprojection(y_fracs, output_folder):
    """Plot the eigen projections."""
    for i, y_frac in enumerate(y_fracs):
        plt.figure()
        plt.hist(y_frac, bins=50)
        plt.xlabel('Eigenvalue')
        plt.ylabel('Frequency')
        plt.title('Eigenspectrum of Temporally Conserved Taxon-Taxon Covariance Matrix')
        plt.savefig(os.path.join(output_folder, f'eigenspectrum_{i + 1}.png'), dpi=300)
        plt.close()

def identify_important_species(ev_wtss, window_size, threshold):
    """Identify important species."""
    important_species = set()
    num_windows = len(ev_wtss) - window_size + 1
    species_importance_values = np.zeros(ev_wtss[0].shape)

    for i in range(num_windows):
        window_ev_wts = ev_wtss[i:i + window_size]
        avg_ev_wts = np.mean(window_ev_wts, axis=0)
        important_indices = np.argsort(avg_ev_wts)[-threshold:]
        important_species.update(important_indices)
        species_importance_values[important_indices] += avg_ev_wts[important_indices]

    return list(important_species), species_importance_values

def main():
    """Main function to execute the workflow."""
    data1_path = '<path_to_data1>'
    data2_path = '<path_to_data2>'
    timeframe = [3, 12] #user input
    threshold = 10 #user input
    eigenvector_count = 1#user input
    window_size = 3#user input

    try:
        data1, data2, OTU_ID, OTU_names = load_data(data1_path, data2_path)
        data1_trm, data2_trm = trim_data(data1, data2, timeframe)
        cov_data, reordered_indices, cov_data_reordered = calculate_covariance_and_reorder(data1_trm, data2_trm)
        unique_taxa_indices, taxa_names, OTU_ID_trm, cov_data_uniq = select_taxa(reordered_indices, threshold, OTU_names, OTU_ID, cov_data)
        cov_data_uniq_norm = normalize_cov_data(cov_data_uniq)
        y_fracs, ev_wtss = perform_eigendecomposition(cov_data_uniq_norm, eigenvector_count)
        important_species, species_importance_values = identify_important_species(ev_wtss, window_size, threshold)
if __name__ == "__main__":
    main()
