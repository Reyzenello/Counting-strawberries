import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# Define a function to convert text to a tensor of one-hot encoded characters
def text_to_tensor(text, char_to_index, max_length):
    tensor = torch.zeros(max_length, len(char_to_index))
    for i, char in enumerate(text):
        if i < max_length and char in char_to_index:
            tensor[i][char_to_index[char]] = 1
    return tensor

# Modify the dataset generation to process characters sequentially
def generate_dataset(words, char_to_index, max_length):
    inputs = []
    targets = []
    for word in words:
        word_tensor = text_to_tensor(word, char_to_index, max_length)
        for i in range(len(word)):
            inputs.append(word_tensor[:i + 1])  # Input is the sequence up to the current char
            target = torch.zeros(len(char_to_index))
            for char in word[:i + 1]:
                if char in char_to_index:
                    target[char_to_index[char]] += 1
            targets.append(target)
    return inputs, targets

# Define the RNN model with GRU
class CharacterCountModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(CharacterCountModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None):
        batch_size = x.size(0)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Parameters
hidden_dim = 128
num_epochs = 5000
learning_rate = 0.001
batch_size = 32

# Expanded example words
words = [
    "strawberry", "apple", "banana", "cherry", "blueberry", "raspberry", "blackberry", "kiwi", "grape", "orange",
    "watermelon", "pineapple", "mango", "papaya", "peach", "plum", "pear", "pomegranate", "lemon", "lime",
    "apricot", "avocado", "coconut", "fig", "guava", "jackfruit", "lychee", "nectarine", "passion fruit", "tangerine"
]

# Create character mappings
alphabet = "abcdefghijklmnopqrstuvwxyz"
char_to_index = {char: idx for idx, char in enumerate(alphabet)}

# Determine the maximum length of the words
max_length = max(len(word) for word in words)

# Generate dataset
inputs, targets = generate_dataset(words, char_to_index, max_length)

# Convert to tensors and pad sequences
inputs = [torch.tensor(np.array(seq)) for seq in inputs]
inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True).float()
targets = torch.stack(targets).float()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Set input and output dimensions
input_dim = len(char_to_index)
output_dim = len(char_to_index)

# Create the model
model = CharacterCountModel(input_dim, hidden_dim, output_dim, num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        batch_inputs = X_train[i:i+batch_size]
        batch_targets = y_train[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss/len(X_train):.4f}, Val Loss: {val_loss.item():.4f}")

print("Training complete.")

# Function to predict the count of characters in a word
def predict_counts(word, model, char_to_index, max_length):
    with torch.no_grad():
        word_tensor = text_to_tensor(word, char_to_index, max_length).unsqueeze(0)
        output = model(word_tensor)
        return output.squeeze().numpy()

# Get user input
word_to_predict = input("Enter the word: ").strip().lower()

# Predict the counts of all characters in the input word
predicted_counts = predict_counts(word_to_predict, model, char_to_index, max_length)

# Print the results
for char, count in zip(alphabet, predicted_counts):
    if count > 0:
        print(f"The predicted number of '{char}' in '{word_to_predict}' is: {int(round(count))}")
