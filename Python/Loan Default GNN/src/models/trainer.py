import torch
import torch.optim as optim
import torch.nn as nn 
from .graphsage import GraphSAGEModel  # Relative import

class GraphTrainer:
    @staticmethod
    def train_model(model, data, epochs=1000, print_interval=10):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        data = data.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            if epoch % print_interval == 0:
                pred = out.argmax(dim=1)
                acc = (pred == data.y).sum().item() / data.y.size(0)
                print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")
        
        return model
