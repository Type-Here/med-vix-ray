import torch
from torch import optim


class EarlyStopperMin:
    """
    A class to implement early stopping for training models.

    Attributes:
        patience (int): Number of consecutive epochs with no improvement after which training stops. Default is 1.
        min_delta (float): Minimum change in validation loss to qualify as an improvement. Default is 1e-4.
    """
    def __init__(self, patience=1, min_delta=1e-4):
        """
        Initializes the EarlyStopper with patience and minimum delta.

        Args:
            patience (int): Number of epochs to wait before stopping. Must be non-negative.
            min_delta (float): Minimum improvement to reset the counter. Must be non-negative.
        """
        if patience < 0:
            raise ValueError("`patience` must be a non-negative integer.")
        if min_delta < 0:
            raise ValueError("`min_delta` must be a non-negative float.")

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_model_state = None

    def early_stop(self, validation_loss, model):
        """
        Determines whether training should stop based on validation loss.

        Args:
            validation_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained. Used to save the best model state if needed.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if validation_loss is None:
            raise ValueError("`validation_loss` must not be None.")

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            # Save the best model state
            self.best_model_state = model.state_dict()
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:

                # Save the best model state
                model.load_state_dict(self.best_model_state)
                print(f"Early stopping triggered. Stopping training.")
                print(f"Best validation loss: {self.min_validation_loss:.4f}")
                print(f"Model state restored.")
                return True
        return False


class EarlyStopper:
    """
    Enhanced early stopping for training models with overfitting prevention.
    """

    def __init__(self, patience=1, min_delta=1e-4, max_train_val_gap=0.2,
                 lr_scheduler=None, monitor='val_loss'):
        """
        Args:
            patience (int): Number of epochs to wait before stopping.
            min_delta (float): Minimum improvement to reset the counter.
            max_train_val_gap (float): Maximum allowed gap between training and validation loss.
            lr_scheduler (Union[torch.optim.lr_scheduler._LRScheduler, CustomLRScheduler, None]): Learning rate scheduler to reduce LR on plateau.
            monitor (str): Metric to monitor ('val_loss', 'f1_score', etc.).
        """
        if patience < 0:
            raise ValueError("`patience` must be a non-negative integer.")
        if min_delta < 0:
            raise ValueError("`min_delta` must be a non-negative float.")

        self.patience = patience
        self.min_delta = min_delta
        self.max_train_val_gap = max_train_val_gap
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_model_state = None
        self.best_metrics = {}
        self.lr_scheduler = lr_scheduler
        self.monitor = monitor
        self.history = {'train_loss': [], 'val_loss': [], 'gap': []}

    def early_stop(self, validation_loss, model, train_loss=None, metrics=None):
        """
        Determines whether training should stop based on validation metrics.

        Args:
            validation_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
            train_loss (float, optional): The current training loss for gap monitoring.
            metrics (dict, optional): Additional metrics to track (F1, accuracy, etc.).

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if validation_loss is None:
            raise ValueError("`validation_loss` must not be None.")

        # Update history
        self.history['val_loss'].append(validation_loss)

        # Check for overfitting via train-val gap
        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
            gap = train_loss - validation_loss
            self.history['gap'].append(gap)

            # Stop if gap indicates significant overfitting
            if gap > self.max_train_val_gap and len(self.history['gap']) > 3:
                print(
                    f"Overfitting detected: train-val gap ({gap:.4f}) exceeds threshold ({self.max_train_val_gap:.4f})")
                model.load_state_dict(self.best_model_state)
                return True

        # Standard validation loss improvement check
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model_state = model.state_dict()

            # Save additional metrics
            if metrics:
                self.best_metrics = metrics.copy()

            # Apply learning rate scheduler if provided
            if self.lr_scheduler:
                self.lr_scheduler.step(validation_loss)

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                # Restore best model
                model.load_state_dict(self.best_model_state)
                print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                print(f"Best validation loss: {self.min_validation_loss:.4f}")
                if self.best_metrics:
                    print("Best metrics:", {k: f"{v:.4f}" for k, v in self.best_metrics.items()})
                return True

        return False


class CustomLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Custom learning rate scheduler that handles different parameter groups
    with different scheduling strategies.
    """
    def __init__(self, optimizer, warmup_epochs=5, patience=5, factor=0.1, min_lr=1e-6, cooldown=2):
        """
        Args:
            optimizer: Optimizer with parameter groups
            warmup_epochs: Number of epochs for linear warmup
            patience: Number of epochs with no improvement before reducing LR
            factor: Factor by which to reduce learning rate
            min_lr: Minimum learning rate
            cooldown: Number of epochs to wait before resuming normal operation
        """
        super().__init__(optimizer)
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.cooldown = cooldown

        # Track learning rates and validation losses
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.history = {i: [] for i in range(len(self.base_lrs))}

    def step(self, val_loss=None):
        """
        Update learning rates based on epoch and validation loss
        """
        self.current_epoch += 1

        # Warmup phase: linearly increase learning rate for new layers
        if self.current_epoch <= self.warmup_epochs:
            self._warmup_step()
            return

        # After warmup: use plateau-based decay if validation loss is provided
        if val_loss is not None:
            self._plateau_step(val_loss)
        else:
            # If no validation loss, apply time-based decay
            self._time_decay_step()

        # Record current learning rates
        for i, group in enumerate(self.optimizer.param_groups):
            self.history[i].append(group['lr'])

    def _warmup_step(self):
        """Linear warmup for first few epochs"""
        ratio = min(1.0, self.current_epoch / self.warmup_epochs)

        for i, group in enumerate(self.optimizer.param_groups):
            # For new layers (classifier and first parameter group), apply full warmup
            if i <= 1:  # Assuming first two groups are new layers
                group['lr'] = self.base_lrs[i] * ratio
            # For fine-tuning layers, use a more conservative warmup
            else:
                group['lr'] = self.base_lrs[i] * (0.5 + 0.5 * ratio)

    def _plateau_step(self, val_loss):
        """Reduce LR when validation loss plateaus"""
        if self.cooldown_counter > 0:
            print(f"[Scheduler] Cooldown active, {self.cooldown_counter} epochs remaining.")
            self.cooldown_counter -= 1
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            # Reset bad epochs counter
            self.num_bad_epochs = 0
            # Start cooldown
            self.cooldown_counter = self.cooldown

            # Apply different reduction factors to different parameter groups
            for i, group in enumerate(self.optimizer.param_groups):
                # Swin backbone gets gentler reduction
                factor = self.factor * 1.5 if i == 0 else self.factor
                # Calculate new lr
                new_lr = max(group['lr'] * factor, self.min_lr)
                # Apply new learning rate
                group['lr'] = new_lr

    def _time_decay_step(self):
        """Simple time-based decay: decrease every few epochs"""
        if self.current_epoch % 10 == 0:  # Every 10 epochs
            for i, group in enumerate(self.optimizer.param_groups):
                new_lr = max(group['lr'] * self.factor, self.min_lr)
                group['lr'] = new_lr


# ================================== OPTIMIZER ================================== #

def create_optimizer(model, layers_to_unblock, learning_rate_classifier, learning_rate_swin,
                     learning_rate_input=None, optimizer_param=None):
    """
    Create optimizer with different parameter groups for different parts of the model.
    Ensures no duplicate parameters across groups.
    """
    if learning_rate_input is None:
        learning_rate_input = learning_rate_classifier  # Default to same as classifier

    if optimizer_param is not None:
        return optimizer_param

    params_seen = set()

    def filter_new(params):
        """Return only params not already added"""
        new_params = []
        for p in params:
            if id(p) not in params_seen:
                new_params.append(p)
                params_seen.add(id(p))
        return new_params

    optimizer = optim.AdamW([
        {"params": filter_new(model.swin_model.patch_embed.parameters()),
         "lr": learning_rate_input},

        {"params": filter_new(model.classifier.parameters()),
         "lr": learning_rate_classifier},

        {"params": filter_new(model.swin_model.layers[-layers_to_unblock:].parameters()),
         "lr": learning_rate_swin},

        {"params": filter_new([p for n, p in model.named_parameters()
                               if not any(x in n for x in ['swin_model.layers',
                                                           'swin_model.head',
                                                           'swin_model.patch_embed'])
                               ]),
         "lr": learning_rate_classifier}
    ])

    return optimizer
