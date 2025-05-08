import torch.nn as nn


def fine_tune(model, dataset):
    for param in model.parameters():
        param.requires_grad = False

    number_features = model.fc.in_features
    num_outputs = len(dataset.classes)
    model.fc = nn.Linear(number_features, num_outputs)
