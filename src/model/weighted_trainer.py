"""
Weighted Trainer for handling class imbalance
"""

import torch
import torch.nn as nn
from transformers import Trainer
from typing import Optional


class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that applies class weights to handle imbalanced datasets
    """
    
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.args.device)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation with class weights
        
        Args:
            model: The model being trained
            inputs: Dictionary of inputs including labels
            return_outputs: Whether to return outputs along with loss
            num_items_in_batch: Number of items in batch (for newer transformers versions)
        """
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute weighted cross entropy loss
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
