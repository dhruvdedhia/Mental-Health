import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (adjust the path to where your dataset is located)
df = pd.read_csv('Dataset.csv')

# Drop rows with missing values in 'selftext' and 'Label' columns
dataset_clean = df.dropna(subset=['selftext', 'Label'])

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])  # Remove special characters
    return text

# Apply text preprocessing to 'selftext'
dataset_clean['selftext'] = dataset_clean['selftext'].apply(preprocess_text)

# Normalize labels to lowercase
dataset_clean['Label'] = dataset_clean['Label'].str.lower().str.strip()

# Get class distribution
class_distribution = dataset_clean['Label'].value_counts()

# Display the class distribution
print("Class Distribution:\n", class_distribution)

# Plot the distribution for better visualization
class_distribution.plot(kind='bar', color='skyblue')
plt.title('Class Distribution in the Dataset')
plt.xlabel('Class Labels')
plt.ylabel('Number of Samples')
plt.show()
