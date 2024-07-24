from flask import Flask, request, jsonify
import torch
import torch.nn as nn

app = Flask(__name__)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = SimpleNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    data_tensor = torch.tensor(data)
    prediction = model(data_tensor).item()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run()
