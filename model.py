import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int, 
        num_classes: int, 
        filter_sizes: list = [3, 4, 5], 
        num_filters: int = 100, 
        dropout: float = 0.5,
        pretrained_embeddings: torch.Tensor = None,
        freeze_embeddings: bool = True
    ):
        """
        CNN for Sentence Classification (Yoon Kim, 2014)
        
        Args:
            vocab_size: Size of the target vocabulary.
            embed_dim: Dimension of word embeddings (e.g., 300 for Word2Vec).
            num_classes: Number of target output classes.
            filter_sizes: List of kernel widths for n-gram feature extraction.
            num_filters: Number of feature maps per filter size.
            dropout: Dropout probability.
            pretrained_embeddings: Pre-trained Word2Vec weights tensor.
            freeze_embeddings: If True, embeddings remain static; if False, fine-tuned during training.
        """
        super(TextCNN, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)
        
        # Static vs. Fine-tuned (non-static) toggle
        self.embedding.weight.requires_grad = not freeze_embeddings

        # 2. Convolutional Layers (Multiple filter widths)
        # Input shape to Conv1d: (batch_size, embed_dim, seq_len)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim, 
                out_channels=num_filters, 
                kernel_size=k
            ) for k in filter_sizes
        ])

        # 3. Regularization & Output Layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, text_input: torch.Tensor) -> torch.Tensor:
        # text_input shape: (batch_size, seq_len)
        
        # Embedded shape: (batch_size, seq_len, embed_dim)
        embedded = self.embedding(text_input)
        
        # Permute for Conv1d: (batch_size, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)

        # Apply convolutions and max-over-time pooling
        pooled_outputs = []
        for conv in self.convs:
            # Convolution -> ReLU activation
            # Output shape: (batch_size, num_filters, seq_len - kernel_size + 1)
            x = F.relu(conv(embedded))
            
            # Max-over-time pooling over the sequence dimension
            # Output shape: (batch_size, num_filters)
            pooled = F.max_pool1d(x, kernel_size=x.shape[2]).squeeze(2)
            pooled_outputs.append(pooled)

        # Concatenate features from all filter widths: (batch_size, len(filter_sizes) * num_filters)
        feature_vector = torch.cat(pooled_outputs, dim=1)

        # Apply Dropout and Softmax projection layer
        dropped = self.dropout(feature_vector)
        logits = self.fc(dropped)

        return logits


# ==========================================
# End-to-End Pipeline Usage Demonstration
# ==========================================
if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = 10000
    EMBED_DIM = 300
    NUM_CLASSES = 2  # Binary sentiment analysis (e.g., Positive / Negative)
    FILTER_SIZES = [3, 4, 5]
    NUM_FILTERS = 100
    BATCH_SIZE = 16
    SEQ_LEN = 32

    # Mock pre-trained Word2Vec weights
    pretrained_word2vec = torch.randn(VOCAB_SIZE, EMBED_DIM)

    # 1. Initialize Model (Static Embedding Mode)
    model_static = TextCNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        filter_sizes=FILTER_SIZES,
        num_filters=NUM_FILTERS,
        pretrained_embeddings=pretrained_word2vec,
        freeze_embeddings=True
    )

    # 2. Initialize Model (Fine-Tuned / Non-Static Embedding Mode)
    model_finetuned = TextCNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        filter_sizes=FILTER_SIZES,
        num_filters=NUM_FILTERS,
        pretrained_embeddings=pretrained_word2vec,
        freeze_embeddings=False
    )

    # Synthetic batch of tokenized sentences (Batch Size x Max Sequence Length)
    dummy_input = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))

    # Loss function and Optimizer setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_finetuned.parameters(), lr=1e-3)

    # Forward Pass
    logits = model_finetuned(dummy_input)
    print("Output Logits Shape:", logits.shape)  # Expected: [16, 2]

    # Compute loss (dummy labels)
    dummy_labels = torch.randint(low=0, high=NUM_CLASSES, size=(BATCH_SIZE,))
    loss = criterion(logits, dummy_labels)
    
    # Backward Pass
    loss.backward()
    optimizer.step()
    print(f"Training Step Successful. Initial Loss: {loss.item():.4f}")
