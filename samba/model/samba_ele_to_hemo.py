 
import numpy as np

import torch
import torch.nn as nn
from einops import rearrange

from args import cosine_embedding_loss
from nn.temporal_encoder import PerParcelHrfLearning, WaveletAttentionNet
from mambapy.mamba2 import Mamba2, Mamba2Config


class TemporalRNNCompressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_rnn_layers=2, output_time_steps=30, rnn_type='lstm'):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.rnn_type = rnn_type
        # Choose RNN type: LSTM or GRU
        if rnn_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=n_rnn_layers)
            self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=n_rnn_layers)
        elif rnn_type == 'mamba2':
            config = Mamba2Config(d_model=hidden_size, n_layers=2, d_head=hidden_size//4)
            self.encoder = Mamba2(config)
            self.decoder = Mamba2(config)
    
        # Final linear layer to map to output size
        self.fc = nn.Linear(hidden_size, output_size)

        # Output temporal dimension
        self.output_time_steps = output_time_steps

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 12000, input_size)
        Returns:
            Output tensor of shape (batch_size, 30, output_size)
        """
        batch_size = x.size(0)
        x = self.input_proj(x)

        # Encoder: Compress temporal dimension
        if self.rnn_type == 'lstm':
            _, (hidden, _) = self.encoder(x) if isinstance(self.encoder, nn.LSTM) else self.encoder(x)
        elif self.rnn_type == 'mamba2':
            hidden = self.encoder(x)[:, -1, :]


        # Decoder: Upsample latent representation to desired temporal dimension
        if self.rnn_type == 'lstm':
            # Prepare decoder input: Initialize with zeros for the desired output steps
            decoder_input = torch.zeros(batch_size, self.output_time_steps, hidden.size(-1), device=x.device)
            decoder_output, _ = self.decoder(decoder_input, (hidden, torch.zeros_like(hidden)))
        elif self.rnn_type == 'mamba2':
            decoder_input = hidden.unsqueeze(1).repeat(1, self.output_time_steps, 1)
            decoder_output = self.decoder(decoder_input)

        # Apply a final linear layer to each time step
        output = self.fc(decoder_output)

        return output


class SambaEleToHemo(nn.Module):
    """
    Converts electrophysiological (ele) recordings to hemodynamic (hemo) signals.

    This module incorporates a hierarchical model consisting of:
    1. Per-parcel hemodynamic response function (HRF) learning.
    2. A temporal encoder that uses a wavelet attention network to process
       high-frequency electrophysiological data into a compressed representation
       resembling hemodynamic characteristics.
    3. An attention-based graph upsampling network as a spatial decoder to
       accurately map the input into the hemodynamic space.
    """
    def __init__(self, args):
        super(SambaEleToHemo, self).__init__()
        self.args = args
        self.result_list = []
        self.training_losses = []
        self.validation_losses = []

        self.initialize_networks(args)
        self.mse_loss = nn.MSELoss()

    def initialize_networks(self, args):
        """
        Initializes the network components for HRF learning, temporal encoding,
        and spatial decoding.
        """
        
        # Per-parcel hrf-learning
        if args.hrf_arch == 'hrf':
            self.hrf_learning = PerParcelHrfLearning(args)
        elif args.hrf_arch == 'rnn':
            self.hrf_learning = nn.LSTM(200, 128, args.hrf_layers, batch_first=True) #TODO
        
        if args.temporal_arch == 'wavelet':
        # Per-parcel attention-based wavelet dcomposition learning
            self.temporal_encoder = WaveletAttentionNet(args)
        else:
            self.temporal_encoder = TemporalRNNCompressor(200, 128, 200, output_time_steps=15*424, rnn_type=self.args.temporal_arch)
        
        # Over parcels graph decoder/upsampling
        from nn.spatial_decoder import GMWANet  
        self.spatial_decoder = GMWANet(args).to(args.device)

    def forward(self, x_ele, x_hemo, sub_ele, sub_hemo):
        """
        Processes electrophysiological and hemodynamic data through the network.

        Args:
        - x_ele (torch.float32): Electrophysiological data, shape [batch_size, ele_spatial_dim, ele_temporal_dim].
        - x_hemo (torch.float32): Hemodynamic data, shape [batch_size, hemo_spatial_dim, hemo_temporal_dim].
        - sub_ele (list): List of electrophysiological subject indices for loading adjacency matrices.
        - sub_hemo (list): List of hemodynamic subject indices for loading adjacency matrices.

        Returns:
        - x_hemo_hat (torch.float32): Reconstruction of x_hemo.
        - x_ele_hrf (torch.float32): Inferred per-parcel HRFs.
        - alphas (torch.float32): Inferred wavelet attentions.
        - h_att (torch.float32): Attention graph tensor as a squared matrix.
        """
        
        batch_size = x_ele.shape[0]

        # HRF learning 
        x_ele_hrf = self.hrf_learning(x_ele)                            # [10, 200, 12000]  --> [10, 200, 12000]
        if self.args.hrf_arch == 'hrf':
            x_ele_hrf = x_ele_hrf[:,:,:-1]
        # Temporal encoding with wavelet attention
        if self.args.temporal_arch == 'wavelet':
            x_ele_wavelet, alphas = self.temporal_encoder(x_ele_hrf)        #  [10, 200, 12000]  -> [10, 15, 424]
        else:
            x_ele_wavelet = self.temporal_encoder(x_ele_hrf.permute(0, 2, 1)).permute(0, 2, 1)  #  [10, 200, 12000]  -> [10, 200, 30]
            alphas = 0
            x_ele_wavelet = x_ele_wavelet.reshape(10 * 200, -1, 424)

        # Spatial decoding with graph attention
        teacher_forcing_ratio = self.args.ele_to_hemo_teacher_forcing_ratio if self.training else 0.0
        
        x_hemo_hat, h_att = [], []
        if self.args.mc_probabilistic: 
            for i in range(self.args.mc_n_sampling):
                x_hemo_hat_i, h_att_i = self.spatial_decoder(x_ele_wavelet, x_hemo, batch_size, sub_hemo, sub_ele, teacher_forcing_ratio) 
                x_hemo_hat.append(x_hemo_hat_i.unsqueeze(0))
                h_att.append(h_att_i.unsqueeze(0)) 
        else:
            x_hemo_hat, h_att = self.spatial_decoder(
                x_ele_wavelet, x_hemo, batch_size, sub_hemo, sub_ele, teacher_forcing_ratio
            )
            
        if self.args.mc_probabilistic: 
            x_hemo_hat = torch.cat(x_hemo_hat, dim=0)
            h_att = torch.cat(h_att, dim=0)
        
            x_hemo_hat_std = torch.std(x_hemo_hat, dim=0)
            h_att_std = torch.std(h_att, dim=0)
            
            x_hemo_hat = torch.mean(x_hemo_hat, dim=0)
            h_att = torch.mean(h_att, dim=0)
            
        else:
            x_hemo_hat_std, h_att_std = None, None
             
        # Reshape the HRF (will be used for visualization later) 
        x_ele_hrf = x_ele_hrf.view(-1, x_ele.shape[1], x_ele_hrf.shape[-1])

        return x_hemo_hat, x_hemo_hat_std, x_ele_hrf, alphas, h_att, h_att_std

    def loss(self, x_ele, x_hemo, sub_f, sub_m, iteration):
        """
        Calculates and records the loss between predicted and actual hemodynamic data. 
        
        Args:
        - x_ele (torch.Tensor): Electrophysiological data.
        - x_hemo (torch.Tensor): Actual hemodynamic data.
        - sub_f (list): Indices for electrophysiological subjects.
        - sub_m (list): Indices for hemodynamic subjects.
        - iteration (int): Current iteration number (unused in the method but might be useful for logging).

        Returns:
        - loss (torch.Tensor): Computed loss for the current forward pass.
        - x_hemo_hat (torch.Tensor): Predicted hemodynamic data, returned only during validation.
        - x_hemo (torch.Tensor): Ground truth hemodynamic data, returned only during validation.
        - [zm_hrf, h_att, alphas] (list): Additional outputs from the model, returned only during validation.
        """
         
        x_hemo_hat, x_hemo_hat_std, zm_hrf, alphas, h_att, h_att_std = self.forward(x_ele, x_hemo, sub_f, sub_m)
          
 
        loss = cosine_embedding_loss( 
            rearrange(x_hemo_hat, 'b d t -> b t d'), 
            rearrange(x_hemo, 'b d t -> b t d')
        ) 
         
        # Append the loss for later printing 
        if self.training:
            self.training_losses.append(loss.item())
            return loss
        else:
            self.validation_losses.append(loss.item())
            # During validation, also return the predictions,...
            return loss, x_hemo.detach().cpu(), x_hemo_hat.detach().cpu(), [zm_hrf, h_att, alphas] #, [x_hemo_hat_std.detach().cpu(), h_att_std.detach().cpu()]
    
  
   
    def print_results(self, it):
        """
        Prints and logs the average training and validation losses for a given iteration.
        Args:
        - it (int): The current iteration number.
        """
        
        # Calculate average losses  
        tr_cossloss_average = sum(self.training_losses) / len(self.training_losses) if self.training_losses else 0
        va_cossloss_average = sum(self.validation_losses) / len(self.validation_losses) if self.validation_losses else 0

        # Clear the lists 
        self.training_losses, self.validation_losses = [], []

        # Print
        result = f'itr.: {it:5d} loss (train/valid): {tr_cossloss_average:.3f}/{va_cossloss_average:.3f}'
        print(result)

        # Append the result
        self.result_list.append(result)
