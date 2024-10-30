import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, hamming_loss, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torch.utils.data import Dataset, DataLoader
import psutil
import gc
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold

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

# --------------------- Set Device Globally --------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------- Define Constants and Architectures --------------------- #

# Define the train-test split fraction
train_fraction = 0.9

# Learning rate variable (you can change this value)
learning_rate = 0.001  # Adjust the learning rate here

# Define a list of architectures with varying number of layers and neurons
architectures = [
    {'id': 'arch_1', 'layers': [2048, 512, 216]},         # Architecture 1
    {'id': 'arch_2', 'layers': [2048, 1024, 512]},         # Architecture 2
    {'id': 'arch_3', 'layers': [2048, 1024, 512, 216]},    # Architecture 3
    {'id': 'arch_4', 'layers': [4096, 2048, 1024, 512, 216], 'dropout': True},   # Architecture 4 with Dropout
    {'id': 'arch_5', 'layers': [4096, 2048, 1024, 512, 216], 'elasticnet': True} # Architecture 5 with ElasticNet
]

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
    def __init__(self, layers, dropout=False):
        super(PolytopeModel, self).__init__()
        layer_list = []
        input_dim = 15
        for neurons in layers:
            layer_list.append(nn.Linear(input_dim, neurons))
            layer_list.append(nn.BatchNorm1d(neurons))
            layer_list.append(nn.ReLU())
            if dropout:
                layer_list.append(nn.Dropout(p=0.3))
            input_dim = neurons
        layer_list.append(nn.Linear(input_dim, 46))
        layer_list.append(nn.Sigmoid())
        self.network = nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.network(x)

# --------------------- Model Evaluation Function --------------------- #

def evaluate_models(x_test, y_test):
    """
    Evaluates all saved models on the test set and generates classification reports and Hamming loss.
    Also generates ROC and Precision-Recall curves for each label.

    Args:
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
    """
    # Ensure test data is properly formatted
    if not isinstance(x_test, np.ndarray) or not isinstance(y_test, np.ndarray):
        raise ValueError("x_test and y_test must be NumPy arrays.")

    # Find all saved models
    model_paths = glob.glob("saved_models/model_*.pt")
    if not model_paths:
        print("No saved models found in 'saved_models' directory.")
        return

    # Open the file to save evaluation results (in 'w' mode to erase previous contents)
    with open('evaluation_results.txt', 'w') as file:
        file.write("Model Evaluation Results\n")
        file.write("=" * 40 + "\n")

        # Create test dataset and dataloader
        test_dataset = PolytopeDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=192, shuffle=False)

        # Iterate through each model
        for model_path in model_paths:
            # Extract architecture_id and fold_no from filename
            match = re.search(r'model_(arch_\d+)_fold_(\d+)\.pt', os.path.basename(model_path))
            if not match:
                print(f"Filename {model_path} does not match the expected pattern. Skipping.")
                continue
            architecture_id, fold_no = match.groups()

            print(f"\nEvaluating {model_path}...")
            file.write(f"\nEvaluating model: {model_path}\n")

            try:
                # Determine the architecture based on architecture_id
                architecture_layers = None
                dropout = False
                elasticnet = False
                for arch in architectures:
                    if arch['id'] == architecture_id:
                        architecture_layers = arch['layers']
                        dropout = arch.get('dropout', False)
                        elasticnet = arch.get('elasticnet', False)
                        break
                if architecture_layers is None:
                    print(f"Architecture {architecture_id} not found.")
                    continue

                # Load the model
                model = PolytopeModel(architecture_layers, dropout=dropout)
                model.load_state_dict(torch.load(model_path))
                model.to(device)
                model.eval()

                # Make predictions on the test set
                all_outputs = []
                all_targets = []
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs = inputs.to(device)
                        outputs = model(inputs)
                        all_outputs.append(outputs.cpu().numpy())
                        all_targets.append(targets.numpy())

                y_pred_prob = np.vstack(all_outputs)
                y_true = np.vstack(all_targets)
                y_pred = (y_pred_prob >= 0.5).astype(int)

                # Generate classification report
                report = classification_report(y_true, y_pred, target_names=[f'Class {i}' for i in range(46)], zero_division=0)
                print(f"Classification Report for {model_path}:\n{report}")
                file.write(f"Classification Report for {architecture_id}, Fold {fold_no}:\n{report}\n")

                # Compute Hamming Loss
                hl = hamming_loss(y_true, y_pred)
                print(f"Hamming Loss for {model_path}: {hl}")
                file.write(f"Hamming Loss for {architecture_id}, Fold {fold_no}: {hl}\n")

                # Generate ROC and Precision-Recall curves
                generate_curves(y_true, y_pred_prob, architecture_id, fold_no)

            except Exception as e:
                print(f"An error occurred while evaluating {model_path}: {e}")
                traceback.print_exc()
                file.write(f"An error occurred while evaluating {model_path}: {e}\n")
                file.write(traceback.format_exc())
            
            file.write("=" * 40 + "\n")

    print("\nAll model evaluations have been saved to 'evaluation_results.txt'.")

# Function to generate ROC and Precision-Recall curves for each label
def generate_curves(y_true, y_pred_prob, architecture_id, fold_no):
    num_classes = y_true.shape[1]
    roc_auc_dict = {}
    pr_auc_dict = {}

    # Directories to save plots
    roc_dir = f"roc_curves/{architecture_id}_fold_{fold_no}"
    pr_dir = f"pr_curves/{architecture_id}_fold_{fold_no}"
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(pr_dir, exist_ok=True)

    for i in range(num_classes):
        # Skip labels with only one class present in y_true
        if np.sum(y_true[:, i]) == 0 or np.sum(y_true[:, i]) == y_true.shape[0]:
            print(f"Skipping label {i} due to lack of positive or negative samples.")
            continue

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_dict[f'Class {i}'] = roc_auc

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve for Class {i}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig(f"{roc_dir}/roc_class_{i}.png")
        plt.close()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_prob[:, i])
        pr_auc = average_precision_score(y_true[:, i], y_pred_prob[:, i])
        pr_auc_dict[f'Class {i}'] = pr_auc

        plt.figure()
        plt.plot(recall, precision, label=f'PR curve (AP = {pr_auc:.2f})')
        plt.title(f'Precision-Recall Curve for Class {i}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
        plt.savefig(f"{pr_dir}/pr_class_{i}.png")
        plt.close()

    # Save AUC scores to a file
    with open(f"{roc_dir}/auc_scores.txt", 'w') as f:
        f.write("ROC AUC Scores:\n")
        for cls, score in roc_auc_dict.items():
            f.write(f"{cls}: {score:.4f}\n")

    with open(f"{pr_dir}/ap_scores.txt", 'w') as f:
        f.write("Average Precision Scores:\n")
        for cls, score in pr_auc_dict.items():
            f.write(f"{cls}: {score:.4f}\n")

# --------------------- Main Function --------------------- #

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            # Remove file or link
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # Remove directory
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def main():
    # Clear the contents of the training_scores.txt file
    with open('training_scores.txt', 'w') as f:
        f.write("Training Scores Log\n")
        f.write("=" * 40 + "\n")
    print("Cleared previous training scores.")

    # Ensure the directories exist
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("loss_plots", exist_ok=True)
    os.makedirs("roc_curves", exist_ok=True)
    os.makedirs("pr_curves", exist_ok=True)
    print("Ensured required directories exist.")

    # Clear existing files in the directories
    clear_directory('saved_models')
    clear_directory('loss_plots')
    clear_directory('roc_curves')
    clear_directory('pr_curves')
    print("Cleared previous saved models and loss plots.")

    # Print version information
    print_versions()

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
    entangled_df = final_df[final_df["Separable"] == False].copy()

    # Generate local pie chart
    ml_functions.py.generate_local_pie_chart(entangled_df, save_path="local_graph")

    # Separate the valid training data and calculate coefficients
    valid_entangled_df = entangled_df.dropna(subset=['PolytopeBin']).copy()
    valid_entangled_df['StateCoeff'] = valid_entangled_df['State'].apply(pauli_coefficients)

    # Validate all StateCoeff entries
    if not all(coeff.shape == (15,) for coeff in valid_entangled_df['StateCoeff']):
        raise ValueError("All StateCoeff entries must be of shape (15,)")

    # Create the input and output vectors
    X = np.stack(valid_entangled_df["StateCoeff"].values, axis=0)
    Y = np.stack(valid_entangled_df["PolytopeBin"].values, axis=0)

    # Use MultilabelStratifiedShuffleSplit to split into train and test sets
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 - train_fraction, random_state=42)
    train_index, test_index = next(msss.split(X, Y))
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # Analyze label distribution
    def analyze_label_distribution(y, set_name):
        label_counts = np.sum(y, axis=0)
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(label_counts)), label_counts)
        plt.xlabel('Label Index')
        plt.ylabel('Frequency')
        plt.title(f'Label Distribution in {set_name} Set')
        plt.savefig(f'label_distribution_{set_name}.png')
        plt.close()
        print(f"Saved label distribution plot for {set_name} set as 'label_distribution_{set_name}.png'.")

    analyze_label_distribution(y_train, 'Training')
    analyze_label_distribution(y_test, 'Test')

    # Delete unused dataframes to free up RAM
    del final_df
    del entangled_df
    del valid_entangled_df
    gc.collect()

    # Training loop using MultilabelStratifiedKFold
    mskf = MultilabelStratifiedKFold(n_splits=len(architectures), shuffle=True, random_state=42)
    folds = list(mskf.split(x_train, y_train))

    print("\nStarting training for architectures...")

    # Training loop
    for architecture, (train_index, val_index), fold_no in zip(architectures, folds, range(1, len(architectures)+1)):
        architecture_id = architecture['id']
        architecture_layers = architecture['layers']
        dropout = architecture.get('dropout', False)
        elasticnet = architecture.get('elasticnet', False)
        print(f"\nTraining for {architecture_id}, Fold {fold_no}...")

        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Create Datasets
        train_dataset = PolytopeDataset(x_train_fold, y_train_fold)
        val_dataset = PolytopeDataset(x_val_fold, y_val_fold)

        # Create DataLoaders
        batch_size = 192
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Build the model
        model = PolytopeModel(architecture_layers, dropout=dropout).to(device)

        # Define loss function and optimizer
        # Implement class weights to handle label imbalance
        label_counts = np.sum(y_train_fold, axis=0)
        class_weights = 1.0 / (label_counts + 1e-6)  # Avoid division by zero
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.BCELoss(weight=class_weights_tensor)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Use the learning_rate variable

        # Implement early stopping
        early_stopping_patience = 100
        best_f1_score = 0.0
        patience_counter = 0

        # Initialize best_model_state with the initial model state
        best_model_state = model.state_dict()

        num_epochs = 500
        train_losses = []
        val_losses = []
        f1_train_scores = []
        f1_val_scores = []

        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss = 0.0
            train_outputs = []
            train_targets = []
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Add ElasticNet regularization if specified
                if elasticnet:
                    l1_lambda = 1e-5  # Adjust as needed
                    l2_lambda = 1e-5  # Adjust as needed
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                    loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                train_outputs.append(outputs.detach().cpu().numpy())
                train_targets.append(targets.detach().cpu().numpy())

            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            # Compute training F1 score
            train_outputs_np = np.vstack(train_outputs)
            train_targets_np = np.vstack(train_targets)
            train_pred_labels = (train_outputs_np >= 0.5).astype(int)
            f1_train_weighted = f1_score(train_targets_np, train_pred_labels, average='weighted', zero_division=0)
            f1_train_scores.append(f1_train_weighted)

            # Validation phase
            model.eval()
            val_loss = 0.0
            y_val_pred = []
            y_val_true = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

                    y_val_pred.append(outputs.cpu().numpy())
                    y_val_true.append(targets.cpu().numpy())

            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)

            # Compute validation F1 score
            y_val_pred_np = np.vstack(y_val_pred)
            y_val_true_np = np.vstack(y_val_true)
            y_val_pred_labels = (y_val_pred_np >= 0.5).astype(int)

            f1_val_weighted = f1_score(y_val_true_np, y_val_pred_labels, average='weighted', zero_division=0)
            f1_val_scores.append(f1_val_weighted)

            print(f"Epoch {epoch}/{num_epochs}, Training Loss: {train_loss:.4f}, "
                  f"Validation Loss: {val_loss:.4f}, "
                  f"Training Weighted F1 Score: {f1_train_weighted:.4f}, "
                  f"Validation Weighted F1 Score: {f1_val_weighted:.4f}")

            # Print GPU and RAM usage every 50 epochs
            if epoch % 50 == 0:
                print_gpu_memory()
                print_ram_usage()

            # Early stopping based on validation weighted F1 score
            if f1_val_weighted >= best_f1_score:
                best_f1_score = f1_val_weighted
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Load the best model
        model.load_state_dict(best_model_state)

        # Save the model
        model_save_path = f"saved_models/model_{architecture_id}_fold_{fold_no}.pt"
        torch.save(best_model_state, model_save_path)
        print(f"Model for architecture {architecture_id} fold {fold_no} saved at {model_save_path}")

        # Evaluate the model on validation set
        model.eval()
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        y_val_pred = np.vstack(all_outputs)
        y_val_true = np.vstack(all_targets)
        y_val_pred_labels = (y_val_pred >= 0.5).astype(int)

        # Compute F1 score
        final_f1_weighted = f1_score(y_val_true, y_val_pred_labels, average='weighted', zero_division=0)

        # Write the scores to a text file
        with open('training_scores.txt', 'a') as f:
            f.write(f"Architecture: {architecture_id}, Fold: {fold_no}\n")
            f.write(f"Best Weighted F1 Score: {best_f1_score:.4f}, Final Weighted F1 Score: {final_f1_weighted:.4f}\n")
            f.write("-" * 40 + "\n")

        print(f"Fold {fold_no} - Architecture {architecture_id} - Best Weighted F1 Score: {best_f1_score:.4f}, Final Weighted F1 Score: {final_f1_weighted:.4f}")

        # Plot and save the training and validation loss and F1 score
        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Weighted F1 Score
        plt.subplot(1, 2, 2)
        plt.plot(f1_train_scores, label='Training Weighted F1 Score')
        plt.plot(f1_val_scores, label='Validation Weighted F1 Score')
        plt.title('Weighted F1 Score vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()

        plt.suptitle(f'Performance for {architecture_id} - Fold {fold_no}')
        loss_plot_path = f"loss_plots/performance_{architecture_id}_fold_{fold_no}.png"
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Performance plot saved at {loss_plot_path}")

        # Clear the model and free up GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Cleared model after Fold {fold_no} of {architecture_id}.")

    print("\nAll training processes have completed.")

    # Evaluate all models on the test set and generate classification reports
    evaluate_models(x_test, y_test)

    print("\nEvaluation of all models is complete. Classification reports have been printed.")

# --------------------- Entry Point --------------------- #

if __name__ == "__main__":
    main()
