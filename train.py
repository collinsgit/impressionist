import gc

import torch
import torch.nn as nn
import torch.optim as optim

import dataloader
from models.FeatureModel import FeatureModel


def run():
    num_epochs = 10
    output_period = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FeatureModel()
    model.to(device)

    loss_func = nn.NLLLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    epoch = 1
    while epoch <= num_epochs:
        print('Epoch: ' + str(epoch))

        data_loader = dataloader.get_data('val', {})
        running_loss = 0.

        model.train()

        for batch_num, (images, patches, labels) in enumerate(data_loader, 1):
            images = images.to(device)
            patches = patches.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, patches)
            outputs = outputs.contiguous().view((outputs.size(0), -1))
            loss = loss_func(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            gc.collect()
            if batch_num % output_period == 0:
                print('Loss: {0:.4f}'.format(running_loss / output_period))
                running_loss = 0.

        torch.save(model.state_dict(), 'models/model.{0:}'.format(epoch))
        epoch += 1


if __name__ == '__main__':
    print('Starting Training')
    run()
    print('Finishing Training')
