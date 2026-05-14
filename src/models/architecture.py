import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, vector_dim=384):
        super(NCF, self).__init__()
        # Collaborative layers (User & Item IDs)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP layers to process the fusion of IDs + Review Vector
        input_dim = (embedding_dim * 2) + vector_dim
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user_indices, item_indices, review_vectors):
        u_emb = self.user_embedding(user_indices)
        i_emb = self.item_embedding(item_indices)

        # Concatenate: [User_ID_Vec, Item_ID_Vec, Review_Vec]
        x = torch.cat([u_emb, i_emb, review_vectors], dim=-1)
        return self.fc_layers(x)