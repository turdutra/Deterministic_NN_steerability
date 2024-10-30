import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import psutil
import gc
import optuna
from functools import partial  # For passing additional arguments to Optuna's objective function
import csv  # For writing trial results to CSV files
import seaborn as sns  # For confusion matrix heatmap plotting

import ml_functions.py  # Ensure this module is correctly implemented and accessible

# --------------------- Debugging Functions --------------------- #
def print_gpu_memory():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.1f}MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.1f}MB")
    else:
        print("No GPUs found.")

def print_ram_usage():
    # Get system memory information
    virtual_memory = psutil.virtual_memory()

    # Display RAM usage
    total_memory = virtual_memory.total / (1024 ** 3)  # Convert to GB
    used_memory = virtual_memory.used / (1024 ** 3)    # Convert to GB
    print(f"RAM memory used {used_memory:.2f}/{total_memory:.2f} GB")

# --------------------- Version Information --------------------- #
def print_versions():
    print("----- Environment Versions -----")
    # PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    # CUDA version
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CUDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices found.")
    print("---------------------------------")

# --------------------- Define Constants --------------------- #

# Define the train-test split fraction
train_fraction = 0.9

# --------------------- Utility Functions --------------------- #

# Define a function to calculate Pauli coefficients
def pauli_coefficients(matrix):
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4")

    # Define Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # List of Pauli basis matrices, excluding I ⊗ I
    pauli_basis = [
        np.kron(p1, p2)
        for p1 in [I, X, Y, Z]
        for p2 in [I, X, Y, Z]
        if not (np.array_equal(p1, I) and np.array_equal(p2, I))
    ]

    coefficients = np.zeros(15)  # 16 - 1 = 15 basis matrices
    for i, pauli_matrix in enumerate(pauli_basis):
        coefficients[i] = (np.trace(np.dot(pauli_matrix.conj().T, matrix)) / 4.0).real

    return coefficients

# Define a custom Dataset class
class PolytopeDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# Define the PyTorch model
class PolytopeModel(nn.Module):
    def __init__(self, layers, dropout):
        super(PolytopeModel, self).__init__()
        layer_list = []
        input_dim = 15
        for neurons, drop_p in zip(layers, dropout):
            layer_list.append(nn.Linear(input_dim, neurons))
            layer_list.append(nn.BatchNorm1d(neurons))
            layer_list.append(nn.ReLU())
            if dropout:
                layer_list.append(nn.Dropout(p=drop_p))
            input_dim = neurons
        layer_list.append(nn.Linear(input_dim, 1))
        # Remove the Sigmoid activation
        self.network = nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.network(x).squeeze(1)  # Squeeze to remove extra dimensions

# --------------------- Main Function --------------------- #

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def process_dataset(df, identifier):
    # Clear and create directories
    clear_directory(f'saved_models/{identifier}')
    clear_directory(f'loss_plots/{identifier}')
    clear_directory(f'roc_curves/{identifier}')
    clear_directory(f'pr_curves/{identifier}')
    clear_directory(f'confusion_matrices/{identifier}')
    
    # Proceed with processing...
    print(f"\nProcessing dataset: {identifier}")

    # Extract features and labels
    x = np.stack(df["StateCoeff"].values, axis=0)
    y = df["Local"].values.astype(int)

    # Use StratifiedShuffleSplit for train-test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_fraction, random_state=42)
    for train_indices, test_indices in sss.split(x, y):
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

    # Analyze label distribution
    label_counts = np.bincount(y_train)
    plt.figure(figsize=(6, 4))
    plt.bar(['Non-Local', 'Local'], label_counts)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title(f'Label Distribution in Training Set ({identifier})')
    plt.savefig(f'label_distribution_{identifier}.png')
    plt.close()
    print(f"Saved label distribution plot as 'label_distribution_{identifier}.png'.")

    # Set up CSV file for trial results
    csv_file = f'optuna_trials_{identifier}.csv'
    # If file doesn't exist, write headers
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Trial Number', 'Hyperparameters', 'Validation Metric'])

    # Define the objective function for Optuna
    def objective(trial, csv_file=csv_file):
        try:
            # Hyperparameters
            num_layers = trial.suggest_int('num_layers', 2, 4)
            layers = []
            dropout_rates = []
            for i in range(num_layers):
                num_units = trial.suggest_int(f'n_units_l{i}', 64, 256)
                layers.append(num_units)
                dropout_rate = trial.suggest_float(f'dropout_l{i}', 0.0, 0.5)
                dropout_rates.append(dropout_rate)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

            # Split x_train further into x_train_fold and x_val_fold
            sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=trial.number)
            for train_indices, val_indices in sss_inner.split(x_train, y_train):
                x_train_fold, x_val_fold = x_train[train_indices], x_train[val_indices]
                y_train_fold, y_val_fold = y_train[train_indices], y_train[val_indices]

            # Create Datasets
            train_dataset = PolytopeDataset(x_train_fold, y_train_fold)
            val_dataset = PolytopeDataset(x_val_fold, y_val_fold)

            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=True)

            # Build the model
            model = PolytopeModel(layers, dropout_rates).to(device)

            # Define loss function and optimizer
            # Implement pos_weight to handle class imbalance
            positive_count = np.sum(y_train_fold == 1)
            negative_count = np.sum(y_train_fold == 0)
            pos_weight = negative_count / (positive_count + 1e-6)  # Avoid division by zero
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Training loop
            num_epochs = 50  # Could be adjusted
            best_val_metric = 0.0
            patience_counter = 0
            early_stopping_patience = 10  # Adjust as needed

            for epoch in range(1, num_epochs + 1):
                model.train()
                train_loss = 0.0
                for inputs, targets in train_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device).float()  # Ensure targets are float tensors

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                train_loss = train_loss / len(train_loader.dataset)

                # Validation
                model.eval()
                val_loss = 0.0
                y_val_pred = []
                y_val_true = []
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device)
                        targets = targets.to(device).float()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                        outputs = torch.sigmoid(outputs)
                        y_val_pred.append(outputs.cpu().numpy())
                        y_val_true.append(targets.cpu().numpy())
                val_loss = val_loss / len(val_loader.dataset)
                y_val_pred_np = np.concatenate(y_val_pred, axis=0)
                y_val_true_np = np.concatenate(y_val_true, axis=0)
                y_val_pred_labels = (y_val_pred_np >= 0.5).astype(int)
                val_metric = f1_score(y_val_true_np, y_val_pred_labels, average='binary', zero_division=0)

                # Early stopping
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break  # Early stopping

                # Report intermediate results to Optuna
                trial.report(val_metric, epoch)
                # Handle pruning
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                print(f"Trial {trial.number}, Epoch {epoch}, Val F1 Score: {val_metric:.4f}")

            # Write trial results to CSV file
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([trial.number, trial.params, best_val_metric])

            # Clean up
            del model
            torch.cuda.empty_cache()
            gc.collect()

            return best_val_metric  # Or val_loss if minimizing

        except Exception as e:
            print(f"Exception in trial {trial.number}: {e}")
            # Write exception details to CSV file
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([trial.number, str(trial.params), f"Exception: {e}"])
            raise e  # Re-raise the exception to let Optuna handle it

    # Use partial to pass the csv_file to the objective function
    objective_with_csv = partial(objective, csv_file=csv_file)

    # Create the study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_csv, n_trials=5000)  # Adjust number of trials as needed

    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters for {identifier}: {best_params}")

    # Now, retrain the model with the best hyperparameters on the full training set
    # Build the model with best hyperparameters
    num_layers = best_params['num_layers']
    layers = []
    dropout_rates = []
    for i in range(num_layers):
        num_units = best_params[f'n_units_l{i}']
        layers.append(num_units)
        dropout_rate = best_params[f'dropout_l{i}']
        dropout_rates.append(dropout_rate)
    learning_rate = best_params['learning_rate']
    batch_size = 128

    # Split x_train into x_train_new and x_val
    sss_retrain = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_indices, val_indices in sss_retrain.split(x_train, y_train):
        x_train_new, x_val = x_train[train_indices], x_train[val_indices]
        y_train_new, y_val = y_train[train_indices], y_train[val_indices]

    # Create Datasets and DataLoaders
    train_dataset = PolytopeDataset(x_train_new, y_train_new)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = PolytopeDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build the model
    model = PolytopeModel(layers, dropout_rates).to(device)

    # Define loss function and optimizer
    positive_count = np.sum(y_train_new == 1)
    negative_count = np.sum(y_train_new == 0)
    pos_weight = negative_count / (positive_count + 1e-6)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    # Set the number of epochs to 500
    num_epochs = 500  # Increased from 50 to 500

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'loss_plots/{identifier}/training_validation_loss.png')
    plt.close()
    print(f"Saved loss plot as 'loss_plots/{identifier}/training_validation_loss.png'.")

    # Evaluate the model on the test set
    test_dataset = PolytopeDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float()
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    y_pred_prob = np.concatenate(all_outputs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=['Non-Local', 'Local'], zero_division=0)
    print(f"Classification Report for {identifier}:\n{report}")

    # Compute F1 Score
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    print(f"F1 Score for {identifier}: {f1:.4f}")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot and save the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Local', 'Local'], yticklabels=['Non-Local', 'Local'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {identifier}')
    plt.savefig(f'confusion_matrices/{identifier}/confusion_matrix.png')
    plt.close()
    print(f"Saved confusion matrix as 'confusion_matrices/{identifier}/confusion_matrix.png'.")

    # Save the model
    model_save_path = f"saved_models/{identifier}/best_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_params': best_params,
    }, model_save_path)
    print(f"Best model saved at {model_save_path}")

def main():
    # Ensure the directories exist and clear them
    clear_directory("saved_models")
    clear_directory("loss_plots")
    clear_directory("roc_curves")
    clear_directory("pr_curves")
    clear_directory("confusion_matrices")
    print("Cleared previous saved models and plots.")

    # Print version information
    print_versions()

    # Set device
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the DataFrame from the pickle file with error handling
    try:
        final_df = pd.read_pickle('steering_mp_ds.pkl')
    except FileNotFoundError:
        raise FileNotFoundError("The file 'steering_mp_ds.pkl' was not found. Please ensure it exists in the working directory.")
    except pd.errors.PickleError:
        raise ValueError("The file 'steering_mp_ds.pkl' is not a valid pickle file or is corrupted.")

    # Generate separable pie chart
    ml_functions.py.generate_separable_pie_chart(final_df, save_path="separable_graph")

    # Discard the separable rows and create a copy to avoid SettingWithCopyWarning
    entangled_df = final_df[final_df["Separable"] == False]

    # Generate local pie chart
    ml_functions.py.generate_local_pie_chart(entangled_df, save_path="local_graph")

    # Separate the valid training data and calculate coefficients
    valid_entangled_df = entangled_df.dropna(subset=['Local', 'State']).copy()
    valid_entangled_df['StateCoeff'] = valid_entangled_df['State'].apply(pauli_coefficients)

    # Validate all StateCoeff entries
    if not all(coeff.shape == (15,) for coeff in valid_entangled_df['StateCoeff']):
        raise ValueError("All StateCoeff entries must be of shape (15,)")

    # Process the dataset
    process_dataset(valid_entangled_df, 'entangled')

    # Delete unused dataframes to free up RAM
    del final_df
    del entangled_df
    del valid_entangled_df

    # Invoke garbage collector to free up memory immediately
    gc.collect()

    print("\nAll processes have completed.")

# --------------------- Entry Point --------------------- #

if __name__ == "__main__":
    main()
