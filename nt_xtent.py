import torch
import torch.nn as nn

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask.fill_diagonal_(False)
        for i in range(batch_size):
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False
        return mask

    def forward(self, z_i, z_j):
        """
        Compute the NT-Xent loss in a single-process setting.
        """
        z = torch.cat((z_i, z_j), dim=0)  # Shape: [2*batch_size, dim]
        N = 2 * self.batch_size

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature  # [2N, 2N]

        # Positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # Negative samples
        negative_samples = sim[self.mask].reshape(N, -1)

        # Logits: [positive | negatives]
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=logits.device)

        loss = self.criterion(logits, labels)
        loss /= N
        return loss
