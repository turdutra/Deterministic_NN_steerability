import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import gc
import seaborn as sns  # For confusion matrix heatmap plotting

# --------------------- Utility Functions --------------------- #

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

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
            if drop_p > 0:
                layer_list.append(nn.Dropout(p=drop_p))
            input_dim = neurons
        layer_list.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.network(x).squeeze(1)  # Squeeze to remove extra dimensions

# --------------------- Main Function --------------------- #

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the DataFrame from the pickle file with error handling
    try:
        final_df = pd.read_pickle('steering_mp_ds.pkl')
    except FileNotFoundError:
        raise FileNotFoundError("The file 'steering_mp_ds.pkl' was not found. Please ensure it exists in the working directory.")
    except pd.errors.PickleError:
        raise ValueError("The file 'steering_mp_ds.pkl' is not a valid pickle file or is corrupted.")

    # Discard the separable rows and create a copy to avoid SettingWithCopyWarning
    entangled_df = final_df[final_df["Separable"] == False]

    # Separate the valid training data and calculate coefficients
    valid_entangled_df = entangled_df.dropna(subset=['Local', 'State']).copy()
    valid_entangled_df['StateCoeff'] = valid_entangled_df['State'].apply(pauli_coefficients)

    # Validate all StateCoeff entries
    if not all(coeff.shape == (15,) for coeff in valid_entangled_df['StateCoeff']):
        raise ValueError("All StateCoeff entries must be of shape (15,)")

    # Extract features and labels
    x = np.stack(valid_entangled_df["StateCoeff"].values, axis=0)
    y = valid_entangled_df["Local"].values.astype(int)

    # Define the 10 best architectures directly
    architectures = [
        {
            'num_layers': 4,
            'n_units_l0': 234,
            'dropout_l0': 0.00015725930820056737,
            'n_units_l1': 243,
            'dropout_l1': 0.22547616728692244,
            'n_units_l2': 135,
            'dropout_l2': 0.03728701953238856,
            'n_units_l3': 161,
            'dropout_l3': 0.0744900271940983,
            'learning_rate': 0.004507044968451788
        },
        {
            'num_layers': 4,
            'n_units_l0': 241,
            'dropout_l0': 0.00018179284253259982,
            'n_units_l1': 246,
            'dropout_l1': 0.17041704566488555,
            'n_units_l2': 170,
            'dropout_l2': 0.07829266324577877,
            'n_units_l3': 148,
            'dropout_l3': 0.05565469592317275,
            'learning_rate': 0.0040024670646536895
        },
        {
            'num_layers': 4,
            'n_units_l0': 241,
            'dropout_l0': 0.01000456535608795,
            'n_units_l1': 241,
            'dropout_l1': 0.23314784691652224,
            'n_units_l2': 119,
            'dropout_l2': 0.05355823957440328,
            'n_units_l3': 189,
            'dropout_l3': 0.10977081615340105,
            'learning_rate': 0.004117475094489123
        },
        {
            'num_layers': 4,
            'n_units_l0': 244,
            'dropout_l0': 0.00015983362320492862,
            'n_units_l1': 245,
            'dropout_l1': 0.22621518382123523,
            'n_units_l2': 101,
            'dropout_l2': 0.014155811916717005,
            'n_units_l3': 158,
            'dropout_l3': 0.36887968677339766,
            'learning_rate': 0.0019311822537308645
        },
        {
            'num_layers': 4,
            'n_units_l0': 226,
            'dropout_l0': 0.00018789041928811892,
            'n_units_l1': 252,
            'dropout_l1': 0.21075561923798028,
            'n_units_l2': 94,
            'dropout_l2': 0.04886989161317172,
            'n_units_l3': 164,
            'dropout_l3': 0.09476567425171795,
            'learning_rate': 0.003937837509421164
        },
        {
            'num_layers': 4,
            'n_units_l0': 236,
            'dropout_l0': 0.0003714277139707776,
            'n_units_l1': 238,
            'dropout_l1': 0.17617268450545792,
            'n_units_l2': 135,
            'dropout_l2': 0.04749733616033419,
            'n_units_l3': 148,
            'dropout_l3': 0.08062974871820965,
            'learning_rate': 0.003337478552157527
        },
        {
            'num_layers': 4,
            'n_units_l0': 251,
            'dropout_l0': 0.000790562042625259,
            'n_units_l1': 241,
            'dropout_l1': 0.16379296723617887,
            'n_units_l2': 155,
            'dropout_l2': 0.07947642427608184,
            'n_units_l3': 146,
            'dropout_l3': 0.04806666899931637,
            'learning_rate': 0.0028065022518866927
        },
        {
            'num_layers': 4,
            'n_units_l0': 244,
            'dropout_l0': 0.008586806001884934,
            'n_units_l1': 247,
            'dropout_l1': 0.19777265140628206,
            'n_units_l2': 90,
            'dropout_l2': 0.030918506171551955,
            'n_units_l3': 152,
            'dropout_l3': 0.08990782054303131,
            'learning_rate': 0.004600063607125845
        },
        {
            'num_layers': 4,
            'n_units_l0': 232,
            'dropout_l0': 0.00012714716939388367,
            'n_units_l1': 256,
            'dropout_l1': 0.25012626757300566,
            'n_units_l2': 119,
            'dropout_l2': 0.07023390669157098,
            'n_units_l3': 191,
            'dropout_l3': 0.12066333524786967,
            'learning_rate': 0.003746713304883216
        },
        {
            'num_layers': 4,
            'n_units_l0': 249,
            'dropout_l0': 0.0003317925106808975,
            'n_units_l1': 238,
            'dropout_l1': 0.17913190426233433,
            'n_units_l2': 162,
            'dropout_l2': 0.09072419007252522,
            'n_units_l3': 140,
            'dropout_l3': 0.045429318562442624,
            'learning_rate': 0.0027297207402234864
        }
    ]

    # Prepare for 5-fold cross-validation
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # Create directories to save models and plots
    clear_directory("saved_models")
    clear_directory("loss_plots")
    clear_directory("confusion_matrices")
    print("Cleared previous saved models and plots.")

    # For each architecture
    for arch_idx, hyperparams in enumerate(architectures):
        print(f"\nProcessing Architecture {arch_idx + 1} with hyperparameters: {hyperparams}")

        num_layers = hyperparams['num_layers']
        layers = []
        dropout_rates = []
        for i in range(num_layers):
            n_units_key = f'n_units_l{i}'
            dropout_key = f'dropout_l{i}'
            num_units = hyperparams.get(n_units_key, 128)
            dropout_rate = hyperparams.get(dropout_key, 0.0)
            layers.append(num_units)
            dropout_rates.append(dropout_rate)
        learning_rate = hyperparams.get('learning_rate', 1e-3)

        # For each fold
        for fold_idx, (train_indices, test_indices) in enumerate(skf.split(x, y)):
            print(f"Training fold {fold_idx + 1}/5")

            x_train_fold, x_test_fold = x[train_indices], x[test_indices]
            y_train_fold, y_test_fold = y[train_indices], y[test_indices]

            # Split x_train_fold further into training and validation sets (e.g., 90% train, 10% validation)
            sss_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=fold_idx)
            inner_fold_indices = next(sss_inner.split(x_train_fold, y_train_fold))
            x_train, x_val = x_train_fold[inner_fold_indices[0]], x_train_fold[inner_fold_indices[1]]
            y_train, y_val = y_train_fold[inner_fold_indices[0]], y_train_fold[inner_fold_indices[1]]

            # Create Datasets and DataLoaders
            batch_size = 128
            train_dataset = PolytopeDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            val_dataset = PolytopeDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

            test_dataset = PolytopeDataset(x_test_fold, y_test_fold)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

            # Build the model
            model = PolytopeModel(layers, dropout_rates).to(device)

            # Define loss function and optimizer
            positive_count = np.sum(y_train == 1)
            negative_count = np.sum(y_train == 0)
            pos_weight = negative_count / (positive_count + 1e-6)
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Initialize lists to store losses
            num_epochs = 500
            train_losses = []
            val_losses = []

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

                if epoch % 50 == 0 or epoch == num_epochs:
                    print(f"Epoch {epoch}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # Save the model
            model_save_dir = f"saved_models/architecture_{arch_idx + 1}/fold_{fold_idx + 1}"
            os.makedirs(model_save_dir, exist_ok=True)
            model_save_path = os.path.join(model_save_dir, "best_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparameters': hyperparams,
            }, model_save_path)
            print(f"Model saved at {model_save_path}")

            # Plot the training and validation loss curves
            loss_plot_dir = f"loss_plots/architecture_{arch_idx + 1}"
            os.makedirs(loss_plot_dir, exist_ok=True)
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
            plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Loss Curve for Architecture {arch_idx + 1}, Fold {fold_idx + 1}')
            plt.legend()
            loss_plot_path = os.path.join(loss_plot_dir, f'loss_curve_fold_{fold_idx + 1}.png')
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Loss plot saved at {loss_plot_path}")

            # **Save loss data to file**
            loss_data_path = os.path.join(loss_plot_dir, f'loss_data_fold_{fold_idx + 1}.npz')
            np.savez(loss_data_path, train_losses=train_losses, val_losses=val_losses)
            print(f"Loss data saved at {loss_data_path}")

            # Evaluate the model on the test set
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

            # Compute the confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Plot and save the confusion matrix
            cm_plot_dir = f"confusion_matrices/architecture_{arch_idx + 1}"
            os.makedirs(cm_plot_dir, exist_ok=True)
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Non-Local', 'Local'], yticklabels=['Non-Local', 'Local'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix for Architecture {arch_idx + 1}, Fold {fold_idx + 1}')
            cm_plot_path = os.path.join(cm_plot_dir, f'confusion_matrix_fold_{fold_idx + 1}.png')
            plt.savefig(cm_plot_path)
            plt.close()
            print(f"Confusion matrix saved at {cm_plot_path}")

            # Clean up
            del model
            torch.cuda.empty_cache()
            gc.collect()

    print("\nAll processes have completed.")

# --------------------- Entry Point --------------------- #

if __name__ == "__main__":
    main()
