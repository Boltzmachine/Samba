from argparse import ArgumentParser 
import numpy as np
import torch 
import re

# Importing a specific parcellation class from a module
from data.schaeferparcel_kong2022_17network import SchaeferParcel_Kong2022_17Network

def root_fn(server_mode, dataset):
    dataset = 'eegfmri_translation'
    """
    Generates paths based on the server mode and dataset.
    Raises ValueError if server mode is not recognized.
    """
    # Dictionary to manage dataset paths
    dataset_paths = {
        'eegfmri_translation': 'EEGfMRI/naturalistic_viewing/',
        'megfmri': 'MEGfMRI/forrest_gump/'
    }

    # Dictionary to manage server roots
    server_roots = {
        'misha': '/home/aa2793/scratch/datasets/NEUROSCIENCE/',
        'mccleary': '/home/aa2793/gibbs/arman/datasets/NEUROSCIENCE/'
    }
     
    # Validate dataset and server mode
    if dataset not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset}")
    if server_mode not in server_roots:
        raise ValueError("Specify the server_mode by 'misha', or 'mccleary'")

    # Build paths
    datasets_root = server_roots[server_mode] + dataset_paths[dataset]
    task_dir = 'minute_dataset/' if dataset == 'megfmri' else 'minute_dataset/minute_dataset_translation/'
    graph_dir = 'graphs/' if dataset == 'megfmri' else 'graphs_' + dataset.split('_')[0] + '/' 

    # Full paths for data, graphs, and parcels
    source_data_dir = datasets_root + task_dir
    graph_dir = datasets_root + graph_dir
    parcel_dir = datasets_root + 'schaefer_parcellation_labels/'
    
    return source_data_dir, graph_dir, parcel_dir

def subject_lists(dataset):
    """
    Returns subject lists based on the dataset.
    """
    return (['07'], ['07'])
    # Subject lists for each dataset
    subjects = {
        'megfmri': (['02', '03', '04', '05', '06', '07', '08', '09', '10', '11'],
                    ['02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']),
        'eegfmri_translation': (['07', '08', '10', '11', '12', '13', '14', '16', '19', '20', '21', '22'],
                                ['07', '08', '10', '11', '12', '13', '14', '16', '19', '20', '21', '22'])
    }

    # Validate dataset
    if dataset not in subjects:
        raise ValueError(f"Unknown dataset: {dataset}")

    return subjects[dataset]

import os
def print_gpu_info(args):
    os.system('cls' if os.name == 'nt' else 'clear')
    if torch.cuda.is_available():
        print("---------------------------------------------------")
        print()
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            props = torch.cuda.get_device_properties(device)
            print(f'       Device {i}:  ')
            print(f'           Name: {props.name} ')
            print(f'           Memory: {props.total_memory / 1024 ** 3:.2f} GB')
            print("           ----------------               ") 
            if args.single_subj:
                print('           ' + args.save_prefix)
                
            if args.mc_probabilistic:
                print('           MC-dropout Probabilistic ')
            else:
                print('           accross the subjects!')
            print("           ---------------------------------             ") 
            print(f'           Model: {args.output_key}')
            
            print()
    else:
        print('No GPU available.')

    print("---------------------------------------------------") 
 
def mkdir_fun(path):
    if not os.path.exists(path): 
        os.mkdir(path)
    
def meg2List(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(item + '\n')
    f.close()

def make_directroy(args): 
    
    root_dir = os.path.join('../outputs/'+args.model)
    mkdir_fun(root_dir)
    
    root_dir = root_dir+'/'+ args.output_key
    mkdir_fun(root_dir)
    
    mkdir_fun(os.path.join(root_dir, 'corr'))
    mkdir_fun(os.path.join(root_dir, 'hrf'))
    mkdir_fun(os.path.join(root_dir, 'parcel'))
    mkdir_fun(os.path.join(root_dir, 'xmodel'))   
    mkdir_fun(os.path.join(root_dir, 'graph')) 
    return root_dir  

import torch.nn.functional as F
def cosine_embedding_loss(x, y, reduction='mean'): 
    cos_sim = F.cosine_similarity(x, y)  
    loss = (1 - cos_sim)   
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.mean() 
    return loss

def clean_names(names):
    cleaned_names = []
    prefix = "17networks_"
    for name in names:
        # Decode each element from bytes to string
        decoded_name = name.decode('utf-8')
        
        # Check and remove prefix
        if decoded_name.startswith(prefix):
            decoded_name = decoded_name[len(prefix):]
        
        # Replace underscores with hyphens
        decoded_name = decoded_name.replace('_', '-')
        
        # Remove the trailing "_<number>"
        decoded_name = re.sub(r'(-\d+)$', '', decoded_name)
        
        cleaned_names.append(decoded_name)
    return cleaned_names


def parcel_extractor():
    # Retrieve parcel labels and names  
    source_path = '../dataset/schaefer_parcellation/label_fsaverage5/'
    vertex_200_labels, parcels200_name, ctabs = SchaeferParcel_Kong2022_17Network(source_path, parcel_number=200) 
    parcels200_name.pop(0)  # Remove the background entry 
    parcels200_name.pop(100)
    
    
    vertex_500_labels, names_500, ctabs_500 = SchaeferParcel_Kong2022_17Network(source_path, parcel_number=500)
    
    # params.parcels500_name = [names_500[13:-1] for name in names_500]
    parcels500_name = names_500
    parcels500_name.pop(0)  # Remove the background entry 
    parcels500_name.pop(250)
    
    parcels500_name = clean_names(parcels500_name)
    parcels200_name = clean_names(parcels200_name)
   
    return parcels200_name, parcels500_name, vertex_200_labels, vertex_500_labels

def params_fn(server_mode='misha', dataset='eegfmri_translation'):
    """
    Configures and returns command-line arguments for a neuroscience dataset processing pipeline.
    """
    n_hemo_parcels = 500
    n_ele_parcels = 200

    # Generate file paths based on server mode and dataset
    source_data_dir, graph_dir, parcel_dir = root_fn(server_mode, dataset)

    # Initialize argument parser
    parser = ArgumentParser(description="Setup parameters for neuroscience data processing")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setup arguments
    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--dataset", default=dataset, type=str)
    parser.add_argument("--device", default=device, help='CUDA device')
    parser.add_argument("--graph_dir", default=graph_dir)
    parser.add_argument("--parcels_dir", default=parcel_dir)
    parser.add_argument("--n_hemo_parcels", default=n_hemo_parcels, type=int)
    parser.add_argument("--n_ele_parcels", default=n_ele_parcels, type=int)
    parser.add_argument("--single_subj", default=True, help='If True, dataloader contains only one pair of subjects')
    parser.add_argument("--hemo_adjacency_matrix_dir", default='tbd')
    parser.add_argument("--ele_adjacency_matrix_dir", default='tbd') 
    parser.add_argument("--dropout_rate", default=0.5, help='tbd')
    parser.add_argument("--n_way", default=10, type=int)          #10
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--validation_iteration", default=50, type=int, help='Iteration when the model is validated')
    parser.add_argument("--save_model", default=True, help='Whether to save the model')
    parser.add_argument("--output_key", default='results_key', type=str, help="Folder name for results")
    parser.add_argument("--output_dir", default='your_dir', type=str, help="Directory for saving results")
    parser.add_argument("--hemo_dir", default=f'{source_data_dir}fmri-{n_hemo_parcels}parcell-avg/')
    if dataset == 'eegfmri_translation':
        parser.add_argument("--ele_dir", default=f'{source_data_dir}eeg-{n_ele_parcels}parcell-avg/') 
    else: 
        parser.add_argument("--ele_dir", default=f'{source_data_dir}meg-{n_ele_parcels}parcell-avg/')
    
    
    # SambaEleToHemo parameters 
    parser.add_argument("--ele_to_hemo_n_source_parcels", default=n_ele_parcels, type=int)
    parser.add_argument("--ele_to_hemo_n_target_parcels", default=n_hemo_parcels, type=int) 
    parser.add_argument("--ele_to_hemo_wavelet_dim", default=256, type=int) 
    parser.add_argument("--ele_to_hemo_inverse_time_dim", default=15, type=int)
    parser.add_argument("--ele_to_hemo_in_features", default=15, type=int)       # 30
    parser.add_argument("--ele_to_hemo_n_heads", default=4, type=int)
    parser.add_argument("--ele_to_hemo_dim_head", default=64, type=int)
    parser.add_argument("--ele_to_hemo_n_patches", default=20, type=int)
    parser.add_argument("--ele_to_hemo_lstm_num_layers", default=2, type=int)
    parser.add_argument("--ele_to_hemo_dropout", default=0.3, type=float) 
    parser.add_argument("--ele_to_hemo_teacher_forcing_ratio", default=0.5, type=float) 
    
    
    parser.add_argument("--mc_probabilistic", default=False) 
    parser.add_argument("--mc_dropout", default=0.6, type=float) 
    parser.add_argument("--mc_n_sampling", default=20, type=float) 
    
    # hrf parameters 
    parser.add_argument("--hrf_arch", default="hrf", type=str)
    parser.add_argument("--hrf_length", default=30, type=int)
    parser.add_argument("--hrf_stride", default=1, type=int)  #5 
    parser.add_argument("--hrf_n_parameters", default=6, type=int)
    parser.add_argument("--hrf_temporal_resolution", default=200.0, type=float)
    parser.add_argument("--hrf_response_delay_init", default=6.0, type=float)
    parser.add_argument("--hrf_undershoot_delay_init", default=12.0, type=float)
    parser.add_argument("--hrf_response_dispersion_init", default=0.5, type=float)
    parser.add_argument("--hrf_undershoot_dispersion_init", default=0.7, type=float)
    parser.add_argument("--hrf_response_scale_init", default=0.5, type=float)
    parser.add_argument("--hrf_undershoot_scale_init", default=0.4, type=float)  
    parser.add_argument("--dispersion_deviation", default=0.2, type=float)  
    parser.add_argument("--scale_deviation", default=0.1, type=float)      

    parser.add_argument("--temporal_arch", default="wavelet", type=str)
    parser.add_argument("--wavelet_dims", default=[204, 106, 57, 57])  
    parser.add_argument("--second_translation", default=True)   
    
    # Parse arguments
    params = parser.parse_args()

    parcels200_name, parcels500_name, vertex_200_labels, vertex_500_labels = parcel_extractor()
    
    params.parcels500_name = parcels500_name
    params.parcels200_name = parcels200_name
    
    params.vertex_200_labels = vertex_200_labels
    params.vertex_500_labels = vertex_500_labels
    
    params.lh_rh_lob_names = ['Default', 'Lang.', 'Cont', 'SalVenAttn', 'DorsAttn', 'Aud', 'SomMot', 'Visual']

    # Get subject lists
    ele_sub_list, hemo_sub_list = subject_lists(dataset)
    params.ele_sub_list = ele_sub_list
    params.hemo_sub_list = hemo_sub_list
 
    params.hemo_adjacency_matrix_dir = '../dataset/graphs_eegfmri/fmri_'+str(params.n_hemo_parcels)+'_parcels/'
    if params.dataset=='megfmri':   
        params.ele_adjacency_matrix_dir = '../dataset/graphs_eegfmri/meg_'+str(params.n_ele_parcels)+'_parcels/'   
    elif params.dataset=='eegfmri_translation':   
        params.ele_adjacency_matrix_dir = '../dataset/graphs_eegfmri/eeg_'+str(params.n_ele_parcels)+'_parcels/'
 
    return params













 