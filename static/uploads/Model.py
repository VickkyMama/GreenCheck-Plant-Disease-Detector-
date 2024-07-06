import pandas as pd
import torch.nn as nn


class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine loss
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}  # Combine accuracies

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


# resnet architecture
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  # out_dim : 128 x 64 x 64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)  # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    def forward(self, xb):  # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

idx_to_classes = {0: 'Apple___Apple_scab',
                  1: 'Apple___Black_rot',
                  2: 'Apple___Cedar_apple_rust',
                  3: 'Apple___healthy',
                  # 4: 'Background_without_leaves',
                  4: 'Blueberry___healthy',
                  5: 'Cherry___Powdery_mildew',
                  6: 'Cherry___healthy',
                  7: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
                  8: 'Corn___Common_rust',
                  9: 'Corn___Northern_Leaf_Blight',
                  10: 'Corn___healthy',
                  11: 'Grape___Black_rot',
                  12: 'Grape___Esca_(Black_Measles)',
                  13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                  14: 'Grape___healthy',
                  15: 'Orange___Haunglongbing_(Citrus_greening)',
                  16: 'Peach___Bacterial_spot',
                  17: 'Peach___healthy',
                  18: 'Pepper,_bell___Bacterial_spot',
                  19: 'Pepper,_bell___healthy',
                  20: 'Potato___Early_blight',
                  21: 'Potato___Late_blight',
                  22: 'Potato___healthy',
                  23: 'Raspberry___healthy',
                  24: 'Soybean___healthy',
                  25: 'Squash___Powdery_mildew',
                  26: 'Strawberry___Leaf_scorch',
                  27: 'Strawberry___healthy',
                  28: 'Tomato___Bacterial_spot',
                  29: 'Tomato___Early_blight',
                  30: 'Tomato___Late_blight',
                  31: 'Tomato___Leaf_Mold',
                  32: 'Tomato___Septoria_leaf_spot',
                  33: 'Tomato___Spider_mites Two-spotted_spider_mite',
                  34: 'Tomato___Target_Spot',
                  35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                  36: 'Tomato___Tomato_mosaic_virus',
                  37: 'Tomato___healthy'}
