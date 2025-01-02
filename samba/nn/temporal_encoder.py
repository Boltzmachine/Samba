import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_wavelets import DWT1DForward


class PerParcelHrfLearning(nn.Module): 
    def __init__(self, args):
        super(PerParcelHrfLearning, self).__init__() 
        """
        Initialize a module for per-parcel Hemodynamic Response Function (HRF) learning.
        This module constructs a list of differentiable HRFs tailored to individually learn 
        response and undershooting parameters for each parcel. It then convolves each parcel's 
        neural activity with the inferred HRF.
        
        Args:
            args: The arguments or configurations passed for HRF initialization.
            hrf_stride (int): The stride value for convolution operation.
            n_parcels (int): The total number of parcels to model.
        """
        self.hrf_temporal_resolution = args.hrf_temporal_resolution 
        self.hrf_length = args.hrf_length
        self.hrfs = [Differentiable_HRF(args) for _ in range(args.ele_to_hemo_n_source_parcels)]
             
    def forward(self, x_ele):
        """
        Perform forward propagation through the PerParcelHrfs module by applying learned
        HRFs to the input data. This results in a convolved output representing the 
        hemodynamic response for each parcel.

        Args:
            x_ele (torch.float32): Input data with dimensions  

        Returns:
            zm_hrf (torch.float32): The convolved output of the HRFs for each parcel  
        """
        
        zm_hrf = []
        for p in range(x_ele.shape[1]):
            hrf_p = self.hrfs[p].forward().unsqueeze(0).unsqueeze(0) 
            zm_p_hrf = F.conv1d(x_ele[:, p, :].unsqueeze(1), hrf_p, padding=int((self.hrf_length*self.hrf_temporal_resolution)/2)) 
            zm_hrf.append(zm_p_hrf[:,:,:int(self.hrf_length*self.hrf_temporal_resolution*2)])  
            
        return torch.cat(zm_hrf, dim=1)  
 
def scaled_sigmoid(x, scale, steepness):
    return scale / (1 + torch.exp(-steepness * x))
 
class Differentiable_HRF(nn.Module):
    def __init__(self, args):
        super(Differentiable_HRF, self).__init__()
        """
        Initialize the Differentiable_HRF module designed to learn hemodynamic response functions (HRFs).  

        Args:
            args: A configuration object containing initialization parameters  

        Attributes:
            response_delay, undershoot_delay, response_dispersion, undershoot_dispersion,
            response_scale, undershoot_scale (nn.Parameter): Torch tensors initialized from the args
            and set as trainable parameters.
            net (torch.nn.Sequential): A multi-layer perceptron for inferring updates to HRF parameters.
        """
        # Initialize parameters from the configuration object
        self.hrf_length = args.hrf_length
        self.device = args.device
        self.hrf_temporal_resolution = args.hrf_temporal_resolution
        self.response_delay_init = args.hrf_response_delay_init
        self.undershoot_delay_init = args.hrf_undershoot_delay_init
        self.response_dispersion_init = args.hrf_response_dispersion_init
        self.undershoot_dispersion_init = args.hrf_undershoot_dispersion_init
        self.response_scale_init = args.hrf_response_scale_init
        self.undershoot_scale_init = args.hrf_undershoot_scale_init
        self.dispersion_deviation = args.dispersion_deviation
        self.scale_deviation = args.scale_deviation

        # Define trainable parameters
        self.response_delay = nn.Parameter(torch.tensor(self.response_delay_init).float(), requires_grad=True).to(args.device)
        self.undershoot_delay = nn.Parameter(torch.tensor(self.undershoot_delay_init).float(), requires_grad=True).to(args.device)
        self.response_dispersion = nn.Parameter(torch.tensor(self.response_dispersion_init).float(), requires_grad=True).to(args.device)
        self.undershoot_dispersion = nn.Parameter(torch.tensor(self.undershoot_dispersion_init).float(), requires_grad=True).to(args.device)
        self.response_scale = nn.Parameter(torch.tensor(self.response_scale_init).float(), requires_grad=True).to(args.device)
        self.undershoot_scale = nn.Parameter(torch.tensor(self.undershoot_scale_init).float(), requires_grad=True).to(args.device)

        # Setup the neural network for parameter inference
        hrf_mlp_neurons = [128, 512, 128]
        self.net = torch.nn.Sequential(
            nn.Linear(1, hrf_mlp_neurons[0]),
            # nn.Linear(args.hrf_n_parameters, hrf_mlp_neurons[0]),
            nn.GELU(),
            nn.Linear(hrf_mlp_neurons[0], hrf_mlp_neurons[1]),
            nn.GELU(),
            nn.Linear(hrf_mlp_neurons[1], hrf_mlp_neurons[2]),
            nn.GELU(),
            nn.Linear(hrf_mlp_neurons[2], 1),
            # nn.Linear(hrf_mlp_neurons[2], args.hrf_n_parameters),
        ).to(args.device)
   
    def _double_gamma_hrf(
        self,
        response_delay,
        undershoot_delay,
        response_dispersion,
        undershoot_dispersion,
        response_scale,
        undershoot_scale,
        temporal_resolution,
        ):
        """Create the double gamma HRF with the timecourse evoked activity.
        Default values are based on Glover, 1999 and Walvaert, Durnez,
        Moerkerke, Verdoolaege and Rosseel, 2011
        
        only _double_gamma_hrf is adapted from: 
        https://github.com/brainiak/brainiak/blob/master/brainiak/utils/fmrisim.py

        Parameters
        ----------

        response_delay : float
            How many seconds until the peak of the HRF

        undershoot_delay : float
            How many seconds until the trough of the HRF

        response_dispersion : float
            How wide is the rising peak dispersion

        undershoot_dispersion : float
            How wide is the undershoot dispersion

        response_scale : float
            How big is the response relative to the peak

        undershoot_scale :float
            How big is the undershoot relative to the trough

        scale_function : bool
            Do you want to scale the function to a range of 1

        temporal_resolution : float
            How many elements per second are you modeling for the stimfunction
        Returns
        ----------

        hrf : multi dimensional array
            A double gamma HRF to be used for convolution.

        """ 
        hrf_len = int(self.hrf_length * temporal_resolution)
        hrf_counter = torch.arange(hrf_len).float().to(self.device) 
        
        response_peak = response_delay * response_dispersion
        undershoot_peak = undershoot_delay * undershoot_dispersion

        # Specify the elements of the HRF for both the response and undershoot
        resp_pow = torch.pow((hrf_counter / temporal_resolution) / response_peak, response_delay)
        resp_exp = torch.exp(-((hrf_counter / temporal_resolution) - response_peak) / response_dispersion)

        response_model = response_scale * resp_pow * resp_exp

        undershoot_pow = torch.pow((hrf_counter / temporal_resolution) / undershoot_peak, undershoot_delay)
        undershoot_exp = torch.exp(-((hrf_counter / temporal_resolution) - undershoot_peak) / undershoot_dispersion)

        undershoot_model = undershoot_scale * undershoot_pow * undershoot_exp

        # For each time point, find the value of the HRF
        hrf = response_model - undershoot_model 
        return hrf  
    
    def forward(self, viz_return=False):
        """
        Forward pass to learn six parameters of the Hemodynamic Response Function (HRF)
        using a Multi-Layer Perceptron (MLP). This method involves several steps:
        
        1. Concatenation of differentiable parameters: response delay, undershoot delay,
        response dispersion, undershoot dispersion, response scale, and undershoot scale.
        2. Forward propagation through the MLP to infer updated parameter values.
        3. Application of the double gamma functioparameter_estimationn to separate the inferred parameters into
        their respective components for the next levels of HRF modeling.
        
        Returns:
            torch.Tensor: The inferred HRF modeled with the double gamma function, adjusted
            for the specific temporal resolution of the study.
        """ 

        # Forward pass through the MLP to infer updated parameter values.
        y_hrf_response_delay = self.net(self.response_delay.unsqueeze(0)).squeeze(0)
        y_hrf_undershoot_delay = self.net(self.undershoot_delay.unsqueeze(0)).squeeze(0)
        y_hrf_response_dispersion = self.net(self.response_dispersion.unsqueeze(0)).squeeze(0)
        y_hrf_undershoot_dispersion = self.net(self.undershoot_dispersion.unsqueeze(0)).squeeze(0)
        y_hrf_response_scale = self.net(self.response_scale.unsqueeze(0)).squeeze(0)
        y_hrf_undershoot_scale = self.net(self.undershoot_scale.unsqueeze(0)).squeeze(0) 
          
        steepness = 5 
        response_delay = scaled_sigmoid(y_hrf_response_delay, scale=4, steepness=steepness) + self.response_delay_init
        undershoot_delay = scaled_sigmoid(y_hrf_undershoot_delay, scale=2, steepness=steepness) + self.undershoot_delay_init
         
        response_dispersion = scaled_sigmoid(y_hrf_response_dispersion, scale=.8, steepness=steepness) + self.response_dispersion_init
        undershoot_dispersion = scaled_sigmoid(y_hrf_undershoot_dispersion, scale=0.1, steepness=steepness) + self.undershoot_dispersion_init
 
        response_scale = scaled_sigmoid(y_hrf_response_scale, scale=0.6, steepness=steepness) + self.response_scale_init
        undershoot_scale = scaled_sigmoid(y_hrf_undershoot_scale, scale=0.4, steepness=steepness) + self.undershoot_scale_init
          
        # Compute the HRF using the double gamma function with the inferred parameters.
        hrf_out = self._double_gamma_hrf(
            response_delay=response_delay,
            undershoot_delay=undershoot_delay,
            response_dispersion=response_dispersion,
            undershoot_dispersion=undershoot_dispersion,
            response_scale=response_scale,
            undershoot_scale=undershoot_scale,
            temporal_resolution=self.hrf_temporal_resolution
        )
        
        if viz_return: 
            return hrf_out, [response_delay, undershoot_delay, response_dispersion, undershoot_dispersion, response_scale, undershoot_scale]
        else:
            return hrf_out
    
    
class WaveletAttentionNet(nn.Module):
    def __init__(self, args):
        super(WaveletAttentionNet, self).__init__()
        """
        Initialize the WaveletAttentionNet module which applies wavelet-based encoding with
        a specialized attention mechanism over each frequency band, using the SAMBA
        attention framework. This module transforms neural signal representations
        for improved modeling of hemodynamic responses.

        Args:
            args: A configuration object containing initialization parameters, including
                  dimensions for the wavelet transform, inverse time dimension, and
                  computational device settings.

        Attributes:
            dwt (DWT1DForward): A discrete wavelet transform layer for signal decomposition.
            g1, g2, g3, g4 (nn.Linear): Linear transformation layers for different wavelet bands.
            attention_scores (nn.Linear): Computes attention scores for weighted sum of bands.
            inversewave (ThreeLayerMLP): A multilayer perceptron for signal reconstruction.
        """
        # Device and dimension configuration
        wavelet_dim = args.ele_to_hemo_wavelet_dim
        inverse_time_dim = args.ele_to_hemo_inverse_time_dim
        device = args.device

        # Wavelet transform and band-specific linear transformations
        self.dwt = DWT1DForward(wave='db5', J=3).to(device)
        dims = args.wavelet_dims 
        self.rdn = np.random.RandomState(14)
        self.attention_embedding = ThreeLayerMLP(sum(dims[:]), 128, sum(dims[:])).to(device)
        self.wavelet_attentions = nn.Parameter(
            torch.FloatTensor(self.rdn.standard_normal((1, sum(dims[:])))), 
            requires_grad=True
        )    

    def forward(self, x_meg_hrf):
        """
        Forward pass through the WaveletAttentionNet model to compute joint embeddings of MEG data
        post-hemodynamic response filtering, applying an LSTM autoencoder using a teacher-forced
        algorithm.

        Args:
            x_meg_hrf (torch.float32): The embedding of MEG data after HRF processing.
            x_fmri (torch.float32): Original fMRI data used during training with the teacher-forcing
            ratio; `x_fmri` is used only during training.

        Returns:
            tuple: Contains the output of the LSTM autoencoder and the attention weights.
        """
        n_parcels = x_meg_hrf.shape[1]
        
        # Wavelet transform and individual band processing 
        x_meg_hrf = rearrange(x_meg_hrf, 'b p (m t) -> (b p) m t', m=15)     # m samples per fmri 
        x_wavelet_decomposition = []
        for s in range(x_meg_hrf.shape[1]):  
            xl, xh_list = self.dwt(x_meg_hrf[:, s, :].unsqueeze(1))   
            xh = torch.cat(xh_list, dim=-1)                           
            x_wavelet_decomposition.append(torch.cat((xh, xl), dim=-1))
            
        x_wavelet_decomposition = torch.cat(x_wavelet_decomposition, dim=1)    
        wavelet_attentions = self.attention_embedding(self.wavelet_attentions)
        wavelet_attentions = wavelet_attentions.expand(x_wavelet_decomposition.shape[0], x_wavelet_decomposition.shape[1], -1) 
        
        alphas = F.softmax(wavelet_attentions, dim=-1) 
        x_attentions = x_wavelet_decomposition * alphas 
        return x_attentions, alphas 
    
class ThreeLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayerMLP, self).__init__() 
        
        self.layer1 = nn.Linear(input_size, hidden_size) 
        self.layer2 = nn.Linear(hidden_size, hidden_size) 
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x): 
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x)) 
        x = self.layer3(x)
        return x
    
    