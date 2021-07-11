import time
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import test_dataloader, train_dataloader
from models import FCNs, VGGNet
from torchvision.utils import save_image


def train(epo_num=50, show_vgg_params=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"The model is trained on {device}.")

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)
    fcn_model = fcn_model.to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    # start timing
    start = time.time()
    for epo in range(1, epo_num+1):
        
        train_loss = 0
        fcn_model.train()
        for batch_idx, (batch_data, batch_mask) in enumerate(train_dataloader):
            # batch_data.shape is torch.Size([4, 3, 160, 160])
            # batch_mask.shape is torch.Size([4, 2, 160, 160])

            batch_data, batch_mask = batch_data.to(device), batch_mask.to(device)

            optimizer.zero_grad()
            output = fcn_model(batch_data)
            output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, batch_mask)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        if epo==1 or epo%100==0:
            test_loss = 0
            fcn_model.eval()
            with torch.no_grad():
                for _, (batch_data, batch_mask) in enumerate(test_dataloader):
                    batch_data, batch_mask = batch_data.to(device), batch_mask.to(device)
                    optimizer.zero_grad()

                    output = fcn_model(batch_data)
                    output = torch.sigmoid(output) # output.shape is torch.Size([batch_size, 2, 160, 160])
                    loss = criterion(output, batch_mask)
                    test_loss += loss.item()

                    ground_truth = torch.argmin(batch_mask, dim=1).float().unsqueeze(1)
                    ground_truth = torch.cat([ground_truth, ground_truth, ground_truth], dim=1) # (batchsize, 1, height, width) -> (batchsize, 3, height, width)
                    prediction = torch.argmin(output, dim=1).float().unsqueeze(1)
                    prediction = torch.cat([prediction, prediction, prediction], dim=1)
                    save_image([batch_data[0], ground_truth[0], prediction[0], batch_data[1], ground_truth[1], prediction[1]], f"results/epoch_{epo}.png", nrow=3)

            end = time.time()
            time_used = (end - start)/60
            time_remaning = time_used/(epo)*(epo_num+1-epo)
            print(f"{epo}/{epo_num}, {time_used:.3f} minutes used, {time_remaning:.3f} minutes remaining, train loss = {train_loss/len(train_dataloader):.3f}, test loss = {test_loss/len(test_dataloader):.3f}")
            torch.save(fcn_model, 'results/trained-model.pt')


if __name__ == "__main__":

    train(epo_num=300, show_vgg_params=False)
