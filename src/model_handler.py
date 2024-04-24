import torch


class ModelHandler:
    def __init__(self, model, device, optimizer, loss_fn, train_loader, test_loader):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_epochs(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            for inputs, targets in self.train_loader:
                inputs = inputs.float()
                targets = targets.float()
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Loss calculation
                loss = self.loss_fn(outputs, targets)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    def train_early_stop(self, patience):
        best_loss = float('inf')
        epochs_no_improve = 0
        epochs = 0
        while True:
            self.model.train()
            for inputs, targets in self.train_loader:
                inputs = inputs.float()
                targets = targets.float()
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Loss calculation
                loss = self.loss_fn(outputs, targets)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_inputs, val_targets in self.test_loader:
                    val_inputs = val_inputs.float().to(self.device)
                    val_targets = val_targets.float().to(self.device)
                    val_outputs = self.model(val_inputs)
                    val_loss += self.loss_fn(val_outputs, val_targets).item()

            val_loss /= len(self.test_loader)
            print(f'Epoch {epochs}: Validation Loss: {val_loss}')
            epochs += 1

            # Early stopping logic
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print('Early stopping!')
                    break

    def eval(self):
        self.model.eval()  # Set the model to evaluation mode

        total_loss = 0
        with torch.no_grad():  # No need to track gradients during evaluation
            for inputs, targets in self.test_loader:
                inputs = inputs.float()
                targets = targets.float()
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(self.test_loader)

    def save(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)

    def load(self, model_save_path):
        self.model.load_state_dict(torch.load(model_save_path))
        self.model.to(self.device)

    def generate_next(self, initial_sequence):
        self.model.eval()

        # Convert the initial sequence from NumPy array to PyTorch tensor
        current_sequence = torch.tensor(
            initial_sequence, dtype=torch.float32).to(self.device)

        generated_sequence = []

        with torch.no_grad():
            # Add batch dimension and send to device
            input_sequence = current_sequence.unsqueeze(0).to(self.device)

            # Generate the next step
            new_step = self.model(input_sequence)

            # Remove batch dimension and convert to NumPy
            return new_step.squeeze(0).cpu().numpy()
