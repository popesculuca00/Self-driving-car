from tqdm import tqdm
import torch
from Training_constants import *
from ConditionalImitationLearningNetwork import ImitationLearningNetwork_Training
from CarlaDatasetLoader import get_dataloaders
from utils import jit_compile_model
import time


def train_step(model, img, speed, criterion, optimizer):
    pass


if __name__ == "__main__":

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    model = ImitationLearningNetwork_Training()
    model = jit_compile_model(model)

    train_dataloader, valid_dataloader = get_dataloaders(train_size=TRAIN_TEST_SPLIT_SIZE,num_workers=NUM_WORKERS)
    num_train_batches = len(train_dataloader)
    num_valid_batches = len(valid_dataloader)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_loss_hist, valid_loss_hist = [], []

    for epoch in range(EPOCHS):
        cnt_train_loss = 0.0
        cnt_valid_loss = 0.0

        progress_bar = tqdm(total=num_train_batches)
        progress_bar.write(f"STARTING EPOCH {epoch+1}/{EPOCHS}")


        for index, (img, speed, target, mask) in enumerate(train_dataloader):
            predictions = model(img, speed)
            predictions = predictions * mask

            loss = criterion(predictions)


        
        progress_bar.write(f"Train loss: {cnt_train_loss}\nValidation loss:{cnt_valid_loss}")
        progress_bar.close()
        progress_bar.write(EPOCH_SEPARATOR)

        train_loss_hist.append( cnt_train_loss )
        valid_loss_hist.append( cnt_valid_loss )
