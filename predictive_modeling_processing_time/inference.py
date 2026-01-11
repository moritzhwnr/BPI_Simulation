import torch
import torch.nn as nn
import gpytorch
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. ARCHITECTURE DEFINITIONS
# (Must match training script exactly)
# ---------------------------------------------------------

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, num_activities, embedding_dim, continuous_dim, hidden_dim, output_dim):
        super().__init__()
        # Embedding layer (padding_idx=0)
        self.embedding = nn.Embedding(num_embeddings=num_activities, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm_input_dim = embedding_dim + continuous_dim
        self.lstm = nn.LSTM(self.lstm_input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, mask=None):
        # x shape: [batch, seq_len, features]
        act_ids = x[:, :, 0].long()
        continuous = x[:, :, 1:]
        
        embedded = self.embedding(act_ids) 
        combined_x = torch.cat([embedded, continuous], dim=2) 
        
        lstm_out, _ = self.lstm(combined_x)
        
        if mask is None:
            features = lstm_out[:, -1, :]
        else:
            seq_lengths = mask.sum(dim=1).long() - 1
            seq_lengths = seq_lengths.clamp(min=0)
            batch_indices = torch.arange(x.size(0), device=x.device)
            features = lstm_out[batch_indices, seq_lengths, :]
            
        return self.linear(features)

class GPLayer(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, feature_dim):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DKLModel(nn.Module):
    def __init__(self, feature_extractor, gp_layer):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = gp_layer
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x, mask):
        features = self.feature_extractor(x, mask)
        features = self.scale_to_bounds(features)
        res = self.gp_layer(features)
        return res

# ---------------------------------------------------------
# 2. ACTIVITY MAPPING RECONSTRUCTION
# ---------------------------------------------------------
# We reconstruct the mapping A->Int based on alphabetical sort, 
# which is how .cat.codes works in the training script.

BPMN_ACTIVITIES = sorted([
    'A_Create Application', 'A_Submitted', 'W_Complete application', 
    'A_Concept', 'A_Accepted', 'O_Create Offer', 'O_Created', 
    'O_Sent (mail and online)', 'W_Call after offers', 'A_Cancelled', 
    'O_Cancelled', 'A_Complete', 'W_Validate application', 'O_Accepted', 
    'A_Validating', 'A_Pending', 'O_Returned', 'W_Handle leads', 'A_Denied'
])

# Map: Name -> Int (1-based index, 0 is padding)
ACT_TO_ID = {act: i+1 for i, act in enumerate(BPMN_ACTIVITIES)}

# ---------------------------------------------------------
# 3. PREPROCESSING FUNCTION
# ---------------------------------------------------------

def preprocess_single_trace(trace_data, scaler_mean, scaler_scale):
    """
    Converts a raw trace (list of dicts) into a PyTorch tensor.
    
    Expected trace_data format:
    {
        "attributes": {"RequestedAmount": float},
        "events": [
            {"concept:name": str, "time:timestamp": str (ISO), "org:resource": str},
            ...
        ]
    }
    """
    events = trace_data['events']
    amount = trace_data['attributes']['RequestedAmount']
    
    # Create DataFrame for easier calc
    df = pd.DataFrame(events)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    
    # 1. Activity ID
    # Use .get() to return 0 (Unknown) if activity not in list
    df['ActivityID'] = df['concept:name'].map(ACT_TO_ID).fillna(0).astype(int)
    
    # 2. Time Features
    df['Prev_Time'] = df['time:timestamp'].shift(1)
    # Fill first NaT with current time so Delta is 0
    df['Prev_Time'] = df['Prev_Time'].fillna(df['time:timestamp'])
    
    df['Delta_Time'] = (df['time:timestamp'] - df['Prev_Time']).dt.total_seconds()
    case_start = df['time:timestamp'].min()
    df['Cumulative_Time'] = (df['time:timestamp'] - case_start).dt.total_seconds()
    
    # 3. Resource Feature
    df['automated_resource'] = (df['org:resource'] == 'User_1').astype(int)
    
    # 4. Log Transforms
    log_delta = np.log1p(df['Delta_Time'].values)
    log_cumul = np.log1p(df['Cumulative_Time'].values)
    log_loan = np.log1p(amount) # Constant for all steps
    
    # 5. Construct Matrix [SeqLen, 5]
    # Cols: [ActivityID, Log_Delta, Log_Cumulative, Log_Loan, Automated_Resource]
    # Note: ActivityID is NOT scaled. Others ARE scaled.
    
    seq_len = len(df)
    tensor_data = np.zeros((seq_len, 5))
    
    tensor_data[:, 0] = df['ActivityID'].values
    
    # scaling: (x - mean) / scale
    # We must match the order of cols_to_scale in training: ['Log_Delta', 'Log_Cumulative', 'log_loan_amount']
    # scaler_mean is array [mean_delta, mean_cumul, mean_loan]
    
    tensor_data[:, 1] = (log_delta - scaler_mean[0]) / scaler_scale[0]
    tensor_data[:, 2] = (log_cumul - scaler_mean[1]) / scaler_scale[1]
    tensor_data[:, 3] = (log_loan - scaler_mean[2]) / scaler_scale[2]
    
    tensor_data[:, 4] = df['automated_resource'].values
    
    # Convert to Tensor and add Batch Dimension [1, SeqLen, 5]
    return torch.tensor(tensor_data, dtype=torch.float32).unsqueeze(0)

# ---------------------------------------------------------
# 4. LOAD & PREDICT
# ---------------------------------------------------------

def load_model(checkpoint_path, device='cpu'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Retrieve Metadata
    vocab_size = checkpoint.get('vocab_size', 20) # Default fallback
    scaler_mean = checkpoint['scaler_mean']
    scaler_scale = checkpoint['scaler_scale']
    
    # Init Architecture
    extractor = LSTMFeatureExtractor(
        num_activities=vocab_size, 
        embedding_dim=5, 
        continuous_dim=4, 
        hidden_dim=16, 
        output_dim=2
    )
    gp_layer = GPLayer(torch.randn(20, 2), 2)
    model = DKLModel(extractor, gp_layer)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    
    model.to(device).eval()
    likelihood.to(device).eval()
    
    return model, likelihood, scaler_mean, scaler_scale

def predict_duration(model, likelihood, input_tensor, scaler_mean, scaler_scale):
    """
    Returns predicted duration in seconds (mean) and 95% CI.
    """
    with torch.no_grad():
        # Mask is all 1s for single inference (no padding needed)
        mask = torch.ones(input_tensor.shape[0], input_tensor.shape[1])
        
        # 1. GP Prediction (Z-scores)
        observed_pred = likelihood(model(input_tensor, mask=mask))
        pred_mean_z = observed_pred.mean
        pred_var_z = observed_pred.variance
        
        # 2. Un-Scale (Log Domain)
        # We only care about Log_Delta (Index 0 of scaler) for the target
        target_mean = scaler_mean[0]
        target_scale = scaler_scale[0]
        
        pred_mean_log = (pred_mean_z * target_scale) + target_mean
        pred_var_log = pred_var_z * (target_scale ** 2)
        
        # 3. Log -> Seconds (LogNormal Mean)
        # E[X] = exp(mu + sigma^2/2) - 1 (because of log1p)
        seconds_mean = torch.expm1(pred_mean_log + (pred_var_log / 2)).item()
        
        # CI
        lower_log = pred_mean_log - 2 * torch.sqrt(pred_var_log)
        upper_log = pred_mean_log + 2 * torch.sqrt(pred_var_log)
        
        seconds_lower = torch.expm1(lower_log).item()
        seconds_upper = torch.expm1(upper_log).item()
        
        return seconds_mean, seconds_lower, seconds_upper

# ---------------------------------------------------------
# 5. MAIN EXECUTION EXAMPLE
# ---------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Define Data (A single incomplete case)
    # This represents a case that has just completed "A_Submitted"
    sample_trace = {
        "attributes": {
            "RequestedAmount": 20000.0
        },
        "events": [
            {
                "concept:name": "A_Create Application", 
                "time:timestamp": "2016-01-01T09:00:00.000+00:00", 
                "org:resource": "User_1" # Automated
            },
            {
                "concept:name": "A_Submitted_custom", 
                "time:timestamp": "2016-01-01T09:30:00.000+00:00", 
                "org:resource": "User_1"
            },
            {
                "concept:name": "W_Handle leads", 
                "time:timestamp": "2016-01-01T10:15:00.000+00:00", 
                "org:resource": "User_1" # Human
            }
        ]
    }
    
    checkpoint_path = 'checkpoints/dkl_best_model.pth'
    
    try:
        # Load
        model, likelihood, s_mean, s_scale = load_model(checkpoint_path)
        
        # Process
        input_tensor = preprocess_single_trace(sample_trace, s_mean, s_scale)
        
        # Predict
        duration, lower, upper = predict_duration(model, likelihood, input_tensor, s_mean, s_scale)
        
        print("\n--- Prediction Result ---")
        print(f"Input Sequence Length: {len(sample_trace['events'])}")
        print(f"Predicted Time to Next Event: {duration:.2f} seconds")
        print(f"Confidence Interval (95%):    {lower:.2f}s - {upper:.2f}s")
        print("-------------------------")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure you have run the training script at least once to generate 'checkpoints/dkl_best_model.pth'.")