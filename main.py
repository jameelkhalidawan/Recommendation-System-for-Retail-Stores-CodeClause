import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import *
from tkinter import messagebox

# Load the dataset with the specified encoding and comma delimiter
data = pd.read_csv('data.csv', encoding='ISO-8859-1', delimiter=',')

# Select relevant columns
data = data[['Customer_ID', 'Product_ID', 'Sales', 'State', 'Product_Name']]

# Preprocess the data (fill missing values, convert to lowercase, etc.)
data['State'] = data['State'].fillna('')
data['Product_Name'] = data['Product_Name'].fillna('')
data['Sales'] = data['Sales'].fillna(0)

# Combine the relevant columns into a single text column for TF-IDF vectorization
data['combined_features'] = data['State'] + ' ' + data['Product_Name']

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the TF-IDF vectorizer on the combined_features column
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])

# Compute the cosine similarity between products
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# Define the Neural Collaborative Filtering (NCF) model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, dropout_prob):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        interaction = torch.cat([user_embedding, item_embedding], dim=-1)
        interaction = self.dropout(interaction)
        x = torch.relu(self.fc1(interaction))
        x = torch.relu(self.fc2(x))
        output = self.output(x)
        return output


# Function to train and evaluate the NCF model
def train_evaluate_ncf(hidden_dim, learning_rate, weight_decay, dropout_prob, patience):
    num_epochs = 50
    batch_size = 64

    # Map customer IDs and product IDs to contiguous integers
    user_mapping = {uid: i for i, uid in enumerate(data['Customer_ID'].unique())}
    item_mapping = {iid: i for i, iid in enumerate(data['Product_ID'].unique())}
    num_users = len(user_mapping)
    num_items = len(item_mapping)

    # Create training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create PyTorch DataLoader for training
    train_users = torch.tensor(train_data['Customer_ID'].map(user_mapping).values, dtype=torch.long)
    train_items = torch.tensor(train_data['Product_ID'].map(item_mapping).values, dtype=torch.long)
    train_sales = torch.tensor(train_data['Sales'].values, dtype=torch.float32)
    train_dataset = TensorDataset(train_users, train_items, train_sales)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create the NCF model
    model = NCF(num_users, num_items, hidden_dim, dropout_prob)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize variables for early stopping
    best_rmse = float('inf')
    early_stopping_counter = 0

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0

        model.train()

        for user, item, sales in train_loader:
            optimizer.zero_grad()
            outputs = model(user, item)
            loss = criterion(outputs, sales.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_users = torch.tensor(test_data['Customer_ID'].map(user_mapping).values, dtype=torch.long)
        val_items = torch.tensor(test_data['Product_ID'].map(item_mapping).values, dtype=torch.long)
        val_sales = torch.tensor(test_data['Sales'].values, dtype=torch.float32)

        with torch.no_grad():
            val_outputs = model(val_users, val_items)
            val_loss = criterion(val_outputs, val_sales.unsqueeze(1))
            val_rmse = torch.sqrt(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f} - Validation RMSE: {val_rmse:.4f}')

        train_losses.append(avg_loss)
        val_losses.append(val_rmse)

        # Early stopping based on patience
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    return best_rmse, train_losses, val_losses


# Define hyperparameter search space
param_space = {
    'hidden_dim': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1],
    'weight_decay': [0.001, 0.01, 0.1],
    'dropout_prob': [0.0, 0.2, 0.5],
    'patience': [5, 10, 15]
}

# Initialize best hyperparameters and best RMSE
best_hyperparameters = None
best_rmse = float('inf')

# Random search for hyperparameter tuning
num_searches = 20
for _ in range(num_searches):
    hyperparameters = {param: np.random.choice(values) for param, values in param_space.items()}

    # Train and evaluate NCF with current hyperparameters
    rmse, _, _ = train_evaluate_ncf(**hyperparameters)

    # Update best hyperparameters if the RMSE improves
    if rmse < best_rmse:
        best_rmse = rmse
        best_hyperparameters = hyperparameters

print('Best Hyperparameters:', best_hyperparameters)

# Train and evaluate the NCF model with the best hyperparameters
best_rmse, train_losses, val_losses = train_evaluate_ncf(**best_hyperparameters)

# Plot training and validation losses
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation RMSE')
plt.xlabel('Epoch')
plt.ylabel('Loss / RMSE')
plt.legend()
plt.title('Training and Validation Losses')
plt.show()


# Function to get product recommendations based on user input (state)
def get_recommendations(state_input, cosine_sim=cosine_sim):
    # Find products that match the input state
    state_matches = data[data['State'].str.lower() == state_input.lower()]

    if state_matches.empty:
        return []

    # Find the product that can generate maximum sales in the state
    max_sales_product = state_matches[state_matches['Sales'] == state_matches['Sales'].max()]['Product_Name'].values[0]

    # Find the index of the product that matches the maximum sales product
    idx = data[data['Product_Name'] == max_sales_product].index[0]

    # Get the pairwise similarity scores of all products with the maximum sales product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 most similar products (excluding the maximum sales product)
    sim_scores = sim_scores[1:6]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Return the top 5 recommended products
    recommendations = data['Product_Name'].iloc[product_indices].values.tolist()

    return recommendations


# Create a basic Tkinter GUI
def recommend_products():
    state_input = state_entry.get()
    recommendations = get_recommendations(state_input)
    if recommendations:
        recommendation_text.set('\n'.join(recommendations))
    else:
        recommendation_text.set('No recommendations found for the entered state.')


def show_graphs():
    # Additional graph plotting code here
    # Example: Plotting a histogram of sales
    plt.figure(figsize=(10, 6))
    plt.hist(data['Sales'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sales')
    plt.show()


root = Tk()
root.title('Retail Store Recommendation System')

frame = Frame(root)
frame.pack(padx=20, pady=20)

state_label = Label(frame, text='Enter State:')
state_label.grid(row=0, column=0)

state_entry = Entry(frame)
state_entry.grid(row=0, column=1)

recommend_button = Button(frame, text='Recommend Products', command=recommend_products)
recommend_button.grid(row=1, columnspan=2, pady=10)

recommendation_text = StringVar()
recommendation_label = Label(frame, textvariable=recommendation_text, wraplength=400, justify='left')
recommendation_label.grid(row=2, columnspan=2, pady=10)

graphs_button = Button(frame, text='Graphs & Insights', command=show_graphs)
graphs_button.grid(row=3, columnspan=2, pady=10)

root.mainloop()
