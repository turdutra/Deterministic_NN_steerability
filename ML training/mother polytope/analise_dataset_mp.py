import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

final_df = pd.read_pickle('C:/Users/10/Documents/Algorithim codes/LHS ML/Part2/ML dataset/steering_mp_ds.pkl')

final_df.info()

final_df=final_df.dropna(subset=['PolytopeBin'])
# Inspect the first few entries in PolytopeBin to understand its structure
print(final_df.head())

# Extract each bit from the binary vector into separate columns
bit_columns = pd.DataFrame(final_df['PolytopeBin'].tolist())

# Compute the correlation matrix of these bit columns
corr_matrix = bit_columns.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Plot the heatmap with the mask
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdYlGn', center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title("Correlation Heatmap of Binary Vector Bits in 'PolytopeBin'")
plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Split the data into two subsets based on the 'Local' column
local_true_df = final_df[final_df['Local'] == True]
local_false_df = final_df[final_df['Local'] == False]

# Extract bit columns for each subset
bit_columns_true = pd.DataFrame(local_true_df['PolytopeBin'].tolist())
bit_columns_false = pd.DataFrame(local_false_df['PolytopeBin'].tolist())

# Compute the correlation matrices
corr_matrix_true = bit_columns_true.corr()
corr_matrix_false = bit_columns_false.corr()

# Determine the color scale limits based on both matrices
vmin = min(corr_matrix_true.min().min(), corr_matrix_false.min().min())
vmax = max(corr_matrix_true.max().max(), corr_matrix_false.max().max())

# Create masks for the upper and lower triangles
mask_upper = np.triu(np.ones_like(corr_matrix_false, dtype=bool), k=0)  # Upper triangle with diagonal
mask_lower = np.tril(np.ones_like(corr_matrix_true, dtype=bool), k=-1)  # Lower triangle without diagonal

# Plot the combined heatmap
plt.figure(figsize=(12, 10))

# Plot the lower triangle (for `Local=True`), excluding the diagonal
sns.heatmap(corr_matrix_true, mask=mask_upper, annot=False, cmap='RdYlGn', center=0, 
            vmin=vmin, vmax=vmax, square=True, linewidths=.5, cbar=False)

# Plot the upper triangle and diagonal (for `Local=False`)
sns.heatmap(corr_matrix_false, mask=mask_lower, annot=False, cmap='RdYlGn', center=0, 
            vmin=vmin, vmax=vmax, square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Add titles and show the plot
plt.title("Combined Correlation Heatmap for 'PolytopeBin' by 'Local' Status (Diagonal for Local=False)")
plt.show()



binary_matrix = np.vstack(final_df['PolytopeBin'])

# Sum the values for each column (bit position) to count the 1's
count_of_ones = binary_matrix.sum(axis=0)

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.bar(range(len(count_of_ones)), count_of_ones)
plt.xlabel('Bit Position')
plt.ylabel('Count of 1s')
plt.title('Count of 1s in Each Bit Position')

# Show the plot
plt.show()



# Split the binary matrix based on the 'Local' column
local_true_matrix = np.vstack(final_df[final_df['Local'] == True]['PolytopeBin'])
local_false_matrix = np.vstack(final_df[final_df['Local'] == False]['PolytopeBin'])

# Sum the values for each bit position (count of 1's) for both Local=True and Local=False
count_of_ones_local_true = local_true_matrix.sum(axis=0)
count_of_ones_local_false = local_false_matrix.sum(axis=0)

# Plot the histograms
bit_positions = range(len(count_of_ones_local_true))
width = 0.35  # width of the bars

plt.figure(figsize=(10, 6))

# Plot Local=True counts in one color
plt.bar(bit_positions, count_of_ones_local_true, width, label='Local=True', color='b')

# Plot Local=False counts in another color, with some offset
plt.bar([p + width for p in bit_positions], count_of_ones_local_false, width, label='Local=False', color='r')

# Add labels and title
plt.xlabel('Bit Position')
plt.ylabel('Count of 1s')
plt.title('Count of 1s in Each Bit Position (Local=True vs Local=False)')

# Add a legend
plt.legend()


# Show the plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

final_df = pd.read_pickle('C:/Users/10/Documents/Algorithim codes/LHS ML/Part2/ML dataset/steering_mp_ds.pkl')

final_df=final_df.dropna(subset=['PolytopeBin'])
# Inspect the first few entries in PolytopeBin to understand its structure
print(final_df.head())

# Extract each bit from the binary vector into separate columns
bit_columns = pd.DataFrame(final_df['PolytopeBin'].tolist())

# Compute the correlation matrix of these bit columns
corr_matrix = bit_columns.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Plot the heatmap with the mask
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdYlGn', center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title("Correlation Heatmap of Binary Vector Bits in 'PolytopeBin'")
plt.show()


binary_matrix = np.vstack(final_df['PolytopeBin'])

# Sum the values for each column (bit position) to count the 1's
count_of_ones = binary_matrix.sum(axis=0)

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.bar(range(len(count_of_ones)), count_of_ones)
plt.xlabel('Bit Position')
plt.ylabel('Count of 1s')
plt.title('Count of 1s in Each Bit Position')

# Show the plot
plt.show()

# Split the binary matrix based on the 'Local' column
local_true_matrix = np.vstack(final_df[final_df['Local'] == True]['PolytopeBin'])
local_false_matrix = np.vstack(final_df[final_df['Local'] == False]['PolytopeBin'])

# Sum the values for each bit position (count of 1's) for both Local=True and Local=False
count_of_ones_local_true = local_true_matrix.sum(axis=0)
count_of_ones_local_false = local_false_matrix.sum(axis=0)

# Plot the histograms
bit_positions = range(len(count_of_ones_local_true))
width = 0.35  # width of the bars

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Local=True counts on the first y-axis
ax1.bar(bit_positions, count_of_ones_local_true, width, label='Unsteerable',color="blue")
ax1.set_xlabel('Bit Position')
ax1.set_ylabel('Count of 1s (Unsteerable)')


# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
ax2.bar([p + width for p in bit_positions], count_of_ones_local_false, width, label='Steerable',color="red")
ax2.set_ylabel('Count of 1s (Steerable)')


# Add a title
plt.title('Count of 1s for Each Bit Position (Unsteerable vs Steerable)')

# Add legends for each y-axis
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()
