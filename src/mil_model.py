import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionPooling, self).__init__()
        self.query = nn.Parameter(torch.randn(feature_dim))

    def forward(self, bag):
        scores = F.softmax(torch.matmul(bag, self.query), dim=0)
        attention_output = torch.sum(scores.unsqueeze(1) * bag, dim=0)
        return attention_output


class MILModel(nn.Module):
    def __init__(
        self,
        feat_extractor: nn.Module,
        feat_dim: int,
        num_classes: int,
        feat_extractor_batch_size: int = 8,
    ):
        super(MILModel, self).__init__()
        feat_extractor_module = list(feat_extractor.children())[:-1]
        self.feat_extractor = torch.nn.Sequential(*feat_extractor_module)
        for param in self.feat_extractor.parameters():
            param.requires_grad = False

        self.feat_extractor_batch_size = feat_extractor_batch_size
        self.feat_dim = feat_dim

        self.attention_pooling = AttentionPooling(feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 3, 512),
            nn.Linear(512, num_classes)
        )

    def forward(self, bags):
        bag_features_list = []

        for bag in bags:
            bag_batches = torch.split(bag, self.feat_extractor_batch_size)
            with torch.no_grad():
                instance_features = torch.zeros((len(bag), self.feat_dim)).to(bag.device)
                for i, batch in enumerate(bag_batches):
                    instance_features[i*8:i*8+len(batch)] = self.feat_extractor(batch).view(-1, self.feat_dim)

            # pooling
            mean_pool = torch.mean(instance_features, dim=0)
            max_pool = torch.max(instance_features, dim=0)[0]
            attention_pool = self.attention_pooling(instance_features)

            # concatenate
            bag_features = torch.cat([mean_pool, max_pool, attention_pool])

            bag_features_list.append(bag_features)

        # stack
        x = torch.stack(bag_features_list)

        # classifier
        x = self.classifier(x)

        return x
