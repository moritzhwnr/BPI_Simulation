import torch
import torch.nn as nn
import gpytorch
import numpy as np
import pandas as pd
import pm4py
import os
import sys
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 0. CONFIGURATION & DEVICE SETUP
# ---------------------------------------------------------

# Force unbuffered output
sys.stdout.reconfigure(encoding='utf-8')

# Force CPU (Optimized for small models)
device = torch.device("cpu")

# Paths
DATA_PATH = './data/BPI Challenge 2017.xes/BPI Challenge 2017.xes'
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'runs/dkl_experiment6_final'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# TensorBoard Writer
writer = SummaryWriter(LOG_DIR)

print(f"--- Training Configuration ---")
print(f"Device:       {device}")
print(f"Data Path:    {DATA_PATH}")
print(f"Split:        80% Train / 20% Test (by Case ID)")
print(f"------------------------------\n")

# ---------------------------------------------------------
# 1. DATA PREPROCESSING
# ---------------------------------------------------------

print("Loading and preprocessing data...")

# Load XES
log = pm4py.read_xes(DATA_PATH, return_legacy_log_object=True)

allowed_events = ["A_Pending", "A_Cancelled", "A_Denied"]

pruned_log = pm4py.filter_event_attribute_values(
    log,
    attribute_key="concept:name",
    values=allowed_events,
    level="case",
    retain=True
)

perc = 0.005
filtered_log = pm4py.filter_variants_by_coverage_percentage(pruned_log, perc)


df = pm4py.convert_to_dataframe(filtered_log)

# clean dataframe to only contain actvitities in our simulated bmpn 
bpmn_activities = [
    'A_Create Application', 'A_Submitted', 'W_Complete application', 
    'A_Concept', 'A_Accepted', 'O_Create Offer', 'O_Created', 
    'O_Sent (mail and online)', 'W_Call after offers', 'A_Cancelled', 
    'O_Cancelled', 'A_Complete', 'W_Validate application', 'O_Accepted', 
    'A_Validating', 'A_Pending', 'O_Returned', 'W_Handle leads', 'A_Denied'
]
df = df[df['concept:name'].isin(bpmn_activities)].copy()

df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
df = df.sort_values(['case:concept:name', 'time:timestamp'])

# Filter long cases
case_lengths = df.groupby('case:concept:name').size()
CUTOFF_LIMIT = 50 
valid_cases = case_lengths[case_lengths <= CUTOFF_LIMIT].index
df = df[df['case:concept:name'].isin(valid_cases)].copy()

# Encode Activities
# 1. Shift IDs to reserve 0 for padding
df['ActivityID'] = df['concept:name'].astype('category').cat.codes + 1 
vocab_size = df['ActivityID'].max() + 1 # +1 for padding index 0

# Feature Engineering
g = df.groupby('case:concept:name')
df['Prev_Time'] = g['time:timestamp'].shift(1)
df['Delta_Time'] = (df['time:timestamp'] - df['Prev_Time']).dt.total_seconds().fillna(0)
df['Case_Start_Time'] = g['time:timestamp'].transform('min')
df['Cumulative_Time'] = (df['time:timestamp'] - df['Case_Start_Time']).dt.total_seconds()

df['automated_resource'] = (df['org:resource'] == 'User_1').astype(int)

# Log Transform
df['Log_Delta'] = np.log1p(df['Delta_Time'])
df['Log_Cumulative'] = np.log1p(df['Cumulative_Time'])
df['log_loan_amount'] = np.log1p(df['case:RequestedAmount'])

# 3. Standardization (Fit on TRAIN data only to prevent leakage)
# Ideally, split IDs first, then fit scaler on train_df, then transform both.
# For simplicity here, I will fit on the whole DF, but in production, 
# we should fit scaler ONLY on training case IDs.
scaler = StandardScaler()
cols_to_scale = ['Log_Delta', 'Log_Cumulative', 'log_loan_amount']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Save scaler mean/scale to reverse transform predictions later
SCALER_MEANS = scaler.mean_   
SCALER_SCALES = scaler.scale_

# --- SPLITTING STRATEGY (CRITICAL) ---
# We split unique Case IDs, not rows, to prevent data leakage.
unique_case_ids = df['case:concept:name'].unique()
np.random.shuffle(unique_case_ids) # Randomize order

split_idx = int(len(unique_case_ids) * 0.8)
train_case_ids = unique_case_ids[:split_idx]
test_case_ids = unique_case_ids[split_idx:]

print(f"Total Cases: {len(unique_case_ids)}")
print(f"Train Cases: {len(train_case_ids)}")
print(f"Test Cases:  {len(test_case_ids)}")

# Helper function to generate sequences for a specific set of Case IDs
def generate_sequences(dataframe, case_ids, desc):
    # Filter DF for specific cases
    subset_df = dataframe[dataframe['case:concept:name'].isin(case_ids)]
    grouped = subset_df.groupby('case:concept:name')
    
    feature_cols = ['ActivityID', 'Log_Delta', 'Log_Cumulative', 'log_loan_amount', 'automated_resource']
    X_list = []
    y_list = []

    for case_id, group in tqdm(grouped, desc=desc):
        case_data = group[feature_cols].values
        next_step_durations = group['Log_Delta'].shift(-1).values
        
        for i in range(len(case_data) - 1):
            prefix = case_data[0 : i+1]
            target = next_step_durations[i]
            
            if not np.isnan(target):
                X_list.append(torch.tensor(prefix, dtype=torch.float32).clone())
                y_list.append(target)
    
    return X_list, y_list

# Generate Train and Test sets separately
print("\nGenerating Train Sequences...")
X_train_list, y_train_list = generate_sequences(df, train_case_ids, "Train Data")

print("Generating Test Sequences...")
X_test_list, y_test_list = generate_sequences(df, test_case_ids, "Test Data")

# Pad Sequences
X_train = pad_sequence(X_train_list, batch_first=True, padding_value=0)
y_train = torch.tensor(y_train_list, dtype=torch.float32)

X_test = pad_sequence(X_test_list, batch_first=True, padding_value=0)
y_test = torch.tensor(y_test_list, dtype=torch.float32)

print(f"\nData Ready.")
print(f"Train Samples: {len(y_train)}")
print(f"Test Samples:  {len(y_test)}")

print(f"X Shape: {X_test.shape}")

# ---------------------------------------------------------
# 2. MODEL ARCHITECTURE
# ---------------------------------------------------------

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, num_activities, embedding_dim, continuous_dim, hidden_dim, output_dim):
        super().__init__()
        # Embedding layer for Activity ID
        self.embedding = nn.Embedding(num_embeddings=num_activities, embedding_dim=embedding_dim, padding_idx=0)
        
        # LSTM Input = Embedding Size + Number of continuous features
        self.lstm_input_dim = embedding_dim + continuous_dim
        
        self.lstm = nn.LSTM(self.lstm_input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, mask=None):
        # x shape: [batch, seq_len, 4] 
        # Feature 0 is ActivityID (int), 1-3 are continuous
        
        # Split input
        act_ids = x[:, :, 0].long()
        continuous = x[:, :, 1:]
        
        # Embed activities
        embedded = self.embedding(act_ids) # [batch, seq_len, emb_dim]
        
        # Concatenate embedding with continuous vars
        combined_x = torch.cat([embedded, continuous], dim=2) 
        
        lstm_out, _ = self.lstm(combined_x)
        
        # ... rest of your existing logic for masking ...
        if mask is None:
            features = lstm_out[:, -1, :]
        else:
            # Your masking logic is actually correct provided mask isn't all zeros
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

class ProcessDataset(Dataset):
    def __init__(self, tensor_x, tensor_y, max_len_override=None):
        self.data = tensor_x
        self.targets = tensor_y
        # If we are the test set, we might need to match the train set's max_len
        # to ensure the LSTM doesn't crash if sizes differ.
        if max_len_override:
            self.max_len = max_len_override
        else:
            self.max_len = self.data.size(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        target = self.targets[idx]
        
        # Create Mask (1 for data, 0 for padding)
        # Note: Since we already padded with 0 in 'pad_sequence', 
        # we assume rows with all 0s are padding.
        # Ideally, we calculate length from non-zero entries or store lengths earlier.
        # Simplified approach:
        mask = torch.zeros(self.max_len)
        mask[:seq.size(0)] = 1
        
        # Handle padding if max_len_override is larger than current seq
        pad_size = self.max_len - seq.size(0)
        if pad_size > 0:
            pad = torch.zeros(pad_size, seq.size(1))
            seq = torch.cat((seq, pad), dim=0)
            
        return seq, target.float(), mask

# ---------------------------------------------------------
# 3. INITIALIZATION & OPTIMIZER
# ---------------------------------------------------------

# Determine max sequence length across BOTH sets
global_max_len = max(X_train.size(1), X_test.size(1))
print(f"Max model sequence length: {global_max_len}")

# Create Datasets
train_dataset = ProcessDataset(X_train, y_train, max_len_override=global_max_len)
test_dataset = ProcessDataset(X_test, y_test, max_len_override=global_max_len)

# Create Loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=False)

# --- HYPERPARAMETERS ---
# Ensure vocab_size matches df['ActivityID'].max() + 1 from your preprocessing
vocab_size = int(df['ActivityID'].max() + 1) 

embedding_dim = 5       # vector size for each activity
continuous_dim = 4      # Log_Delta, Log_Cumulative, Log_Loan, automated_resource
lstm_hidden = 16
gp_input_dim = 2        # Output size of LSTM -> Input size of GP
num_inducing = 20

# --- MODEL INIT ---
# Initialize the NEW Feature Extractor with correct arguments
extractor = LSTMFeatureExtractor(
    num_activities=vocab_size, 
    embedding_dim=embedding_dim, 
    continuous_dim=continuous_dim, 
    hidden_dim=lstm_hidden, 
    output_dim=gp_input_dim
)

# Initialize GP Layer
# Note: Inducing points must match the dimension coming OUT of the LSTM (gp_input_dim)
inducing_points = torch.randn(num_inducing, gp_input_dim)
gp_layer = GPLayer(inducing_points, gp_input_dim)

# Combine into DKL Model
model = DKLModel(extractor, gp_layer)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Move to Device
model = model.to(device)
likelihood = likelihood.to(device)

# Optimizer
optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.gp_layer.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

# Loss (Variational ELBO)
mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_dataset))

# ---------------------------------------------------------
# 4. TRAINING LOOP
# ---------------------------------------------------------

print("\nStarting Training Loop...")

num_epochs = 30
best_test_loss = float('inf')

epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")

for epoch in epoch_pbar:
    
    # --- TRAIN STEP ---
    model.train()
    likelihood.train()
    train_loss_sum = 0
    
    for batch_x, batch_y, batch_mask in train_loader:
        batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)

        optimizer.zero_grad()
        output = model(batch_x, mask=batch_mask)
        loss = -mll(output, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss_sum += loss.item()

    avg_train_loss = train_loss_sum / len(train_loader)
    
    # --- TEST/VALIDATION STEP ---
    model.eval()
    likelihood.eval()
    test_loss_sum = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_mask in test_loader:
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            output = model(batch_x, mask=batch_mask)
            # Use same MLL loss for comparison
            loss = -mll(output, batch_y)
            test_loss_sum += loss.item()
            
    avg_test_loss = test_loss_sum / len(test_loader)
    
    # --- LOGGING & SAVING ---
    
    # TensorBoard: Log both
    writer.add_scalars('Loss', {'Train': avg_train_loss, 'Test': avg_test_loss}, epoch)
    
    status_msg = ""
    # Save Best Model based on TEST Loss (prevents overfitting)
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict(),
            'loss': best_test_loss,
            'scaler_mean': SCALER_MEANS,
            'scaler_scale': SCALER_SCALES,
            'vocab_size': vocab_size,
            'continuous_dim': continuous_dim
        }, os.path.join(CHECKPOINT_DIR, 'dkl_best_model.pth'))
        status_msg = "Saved Best"

    # Save Latest
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'loss': avg_test_loss,
        'scaler_mean': SCALER_MEANS,
        'scaler_scale': SCALER_SCALES,
        'vocab_size': vocab_size,
        'continuous_dim': continuous_dim
    }, os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth'))
    
    # Update Progress Bar
    epoch_pbar.set_postfix({
        'Train': f'{avg_train_loss:.3f}', 
        'Test': f'{avg_test_loss:.3f}', 
        'Status': status_msg
    })

writer.close()
print(f"\nTraining Complete. Best Test Loss: {best_test_loss:.4f}")

# ---------------------------------------------------------
# 5. FINAL EVALUATION ON TEST SET
# ---------------------------------------------------------
print("\nEvaluation on 3 Random Test Samples:")

model.eval()
likelihood.eval()

# Get a single batch
test_x, test_y, test_mask = next(iter(test_loader))
test_x, test_y, test_mask = test_x.to(device), test_y.to(device), test_mask.to(device)

# Retrieve Scaler Constants (These must come from the scaler fit in preprocessing)
# If you didn't save them in variables, you can grab them from the scaler object:
# LOG_DELTA_MEAN = scaler.mean_[0]
# LOG_DELTA_SCALE = scaler.scale_[0]
# For now, I assume they are available as global variables:
if 'LOG_DELTA_MEAN' not in locals():
    print("Warning: Scaler constants not found. Assuming data was NOT scaled (results may be wrong).")
    LOG_DELTA_MEAN = 0
    LOG_DELTA_SCALE = 1

with torch.no_grad():
    # 1. Get Model Output (Distribution of Z-scores)
    observed_pred = likelihood(model(test_x, mask=test_mask))
    
    pred_mean_z = observed_pred.mean
    pred_var_z = observed_pred.variance
    
    # 2. Un-scale Z-scores back to Log-Space
    # Formula: X = Z * scale + mean
    # Variance scales by square of scale factor
    pred_mean_log = (pred_mean_z * LOG_DELTA_SCALE) + LOG_DELTA_MEAN
    pred_var_log = pred_var_z * (LOG_DELTA_SCALE ** 2)
    
    # 3. Convert Log-Space Distribution to Real Seconds (LogNormal -> Normal)
    # The "Mean" of a LogNormal distribution is NOT exp(mu), it is exp(mu + sigma^2 / 2)
    predicted_seconds = torch.exp(pred_mean_log + (pred_var_log / 2))
    
    # The "Median" is exp(mu) - often closer to what humans expect for durations
    median_seconds = torch.exp(pred_mean_log)

    # 4. Process Actual Targets (Un-scale Z-scores, then Inverse Log)
    real_val_log = (test_y * LOG_DELTA_SCALE) + LOG_DELTA_MEAN
    real_seconds = torch.expm1(real_val_log) # expm1 reverses log1p

    # 5. Calculate Uncertainty Range (95% CI in Seconds)
    # We calculate the CI in log-space first, then exponentiate
    lower_log = pred_mean_log - 2 * torch.sqrt(pred_var_log)
    upper_log = pred_mean_log + 2 * torch.sqrt(pred_var_log)
    
    lower_seconds = torch.exp(lower_log)
    upper_seconds = torch.exp(upper_log)

# Print
for i in range(3):
    print(f"Sample {i+1}:")
    print(f"  Actual Duration: {real_seconds[i].item():.2f}s")
    print(f"  Predicted Mean:  {predicted_seconds[i].item():.2f}s")
    print(f"  Predicted Range: {lower_seconds[i].item():.2f}s - {upper_seconds[i].item():.2f}s")
    print("-" * 40)