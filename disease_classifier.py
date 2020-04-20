import torch
import torchvision

class PlantDiseaseClassifier(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        encoder = torchvision.models.resnet18(pretrained=pretrained)
        encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
        for param in encoder.parameters():
            param.requires_grad = False

        self.encoder = encoder
        self.classifier = torch.nn.Sequential(
             torch.nn.Linear(in_features=512, out_features=25),
             torch.nn.Sigmoid()
        )

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            x = self.encoder(x)
            x = torch.flatten(x, 1)
        return self.classifier(x)
