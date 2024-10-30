import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



import matplotlib.patheffects as path_effects

# Function to add black outline to text
def outline_text(text, ax, lw=3):
    for txt in text:
        txt.set_path_effects([plt.Line2D(1, 0), plt.PathEffectStroke(linewidth=lw, foreground='white'), plt.PathEffectNormal()])


# Function to generate pie chart from dataframe
def generate_separable_pie_chart(df, save_path=None):
    
    # Count the occurrences of each state
    count_unsteerable = df['Separable'].fillna('Inconclusive').value_counts()

    # Labels and values for the pie chart
    labels = ['Separable' if x == True else 'Entangled' if x == False else 'Inconclusive' for x in count_unsteerable.index]
    sizes = count_unsteerable.values
    total_entries = len(df)


    colors = ['#ff4d4d', '#3399ff', '#66ff66']

    # Create a pie chart
    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw the pie chart
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})

    # Add the total number of entries at the top
    plt.title(f'Initial number of states: {total_entries}', y=0.96, fontsize=16)

    # Improve the legend
    for text in autotexts:
        text.set_color('black')
        text.set_fontsize(12)
        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    if save_path is not None:
        plt.savefig(f"{save_path}.png", format="png", bbox_inches='tight')
        print(f"Pie chart saved to {save_path}.png")


    # Show the plot
    plt.show()


# Function to generate and optionally save a pie chart from a dataframe
def generate_local_pie_chart(df, save_path=None):
    # Count the occurrences of each state
    count_unsteerable = df['Local'].fillna('Inconclusive').value_counts()

    # Labels and values for the pie chart
    labels = ['Unsteerable' if x == True else 'Steerable' if x == False else 'Inconclusive' for x in count_unsteerable.index]
    sizes = count_unsteerable.values
    total_entries = len(df)

    colors = ['#ff4d4d', '#3399ff', '#66ff66']

    # Create a pie chart
    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw the pie chart
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})

    # Add the total number of entries at the top
    plt.title(f'Initial number of states: {total_entries}', y=0.96, fontsize=16)

    # Improve the legend
    for text in autotexts:
        text.set_color('black')
        text.set_fontsize(12)
        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    # Save the plot if save_path is provided
    if save_path is not None:
        plt.savefig(f"{save_path}.png", format="png", bbox_inches='tight')
        print(f"Pie chart saved to {save_path}.png")
    # Show the plot

    plt.show()


# Function to plot a histogram of the count of 1s in each bit position
def plot_bit_position_histogram(df, save_path=None):
    # Stack the binary matrix from the specified column of the dataframe
    binary_matrix = np.vstack(df['PolytopeBin'])

    # Sum the values for each column (bit position) to count the 1's
    count_of_ones = binary_matrix.sum(axis=0)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(count_of_ones)), count_of_ones)
    plt.xlabel('Bit Position')
    plt.ylabel('Count of 1s')
    plt.title('Count of 1s in Each Bit Position')

    # Save the plot if save_path is provided
    if save_path is not None:
        plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight')
        print(f"Histogram saved to {save_path}.png")

    # Show the plot
    plt.show()


