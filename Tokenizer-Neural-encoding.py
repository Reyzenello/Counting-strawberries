import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def text_to_tensor(text, char_to_index, max_length):
    tensor = torch.zeros(max_length, len(char_to_index))
    for i, char in enumerate(text[:max_length]):
        if char in char_to_index:
            tensor[i][char_to_index[char]] = 1
    return tensor

def generate_dataset(words, char_to_index, max_length):
    inputs = []
    targets = []
    for word in words:
        word_tensor = text_to_tensor(word, char_to_index, max_length)
        target = torch.tensor([word.count(char) for char in char_to_index.keys()])
        inputs.append(word_tensor)
        targets.append(target)
    return torch.stack(inputs), torch.stack(targets)

# Parameters
input_dim = 27  # 26 letters + space
hidden_dim = 128  # Increased hidden dimension
output_dim = 27  # One output per character
num_epochs = 5000  # Increased number of epochs
learning_rate = 0.001
max_length = 15
weight_decay = 1e-5

# Expanded dataset
words = [
    "strawberry", "apple", "banana", "cherry", "blueberry", "raspberry", "blackberry", "kiwi", "grape", "orange",
    "pineapple", "mango", "pear", "peach", "plum", "pomegranate", "watermelon", "nectarine", "lime", "lemon",
    "apricot", "avocado", "blackcurrant", "cantaloupe", "cranberry", "fig", "gooseberry", "grapefruit", "guava",
    "honeydew", "jackfruit", "lychee", "mulberry", "papaya", "passionfruit", "persimmon", "quince", "tangerine",
    "book", "pencil", "paper", "computer", "phone", "table", "chair", "window", "door", "floor", "ceiling",
    "wall", "lamp", "picture", "clock", "calendar", "keyboard", "mouse", "screen", "speaker", "microphone"
]
alphabet = "abcdefghijklmnopqrstuvwxyz "
char_to_index = {char: idx for idx, char in enumerate(alphabet)}

# Prepare the dataset
inputs, targets = generate_dataset(words, char_to_index, max_length)

# Initialize model, criterion, and optimizer
model = LiquidNeuralNetwork(input_dim, hidden_dim, output_dim)
criterion = nn.SmoothL1Loss()  # Huber loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets.float())
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")

# Function to predict the count of a character in a word
def predict_count(word, model, char_to_index, max_length):
    with torch.no_grad():
        word_tensor = text_to_tensor(word, char_to_index, max_length).unsqueeze(0)
        output = model(word_tensor)
        return output.squeeze().round().int()

# User input
word_to_predict = input("Enter the word: ").strip().lower()
char_to_count = input("Enter the character to count: ").strip().lower()

# Predict and display the result
predicted_counts = predict_count(word_to_predict, model, char_to_index, max_length)
char_index = char_to_index.get(char_to_count, -1)
if char_index != -1:
    predicted_count = predicted_counts[char_index].item()
    print(f"The predicted number of '{char_to_count}' in '{word_to_predict}' is: {predicted_count}")
else:
    print(f"Character '{char_to_count}' is not in the alphabet.")
