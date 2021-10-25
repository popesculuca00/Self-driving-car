from tqdm import tqdm
import torch
from Training_constants import *
from ConditionalImitationLearningNetwork import ImitationLearningNetwork_Training, get_learning_rate_scheduler
from CarlaDatasetLoader import get_dataloaders
from utils import jit_compile_model
import time




if __name__ == "__main__":

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    model = ImitationLearningNetwork_Training()
    model = jit_compile_model(model)

    train_dataloader, valid_dataloader = get_dataloaders(train_size=TRAIN_TEST_SPLIT_SIZE,num_workers=NUM_WORKERS) ## todo add drop_last
    num_train_batches = len(train_dataloader)
    num_valid_batches = len(valid_dataloader)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler_lr = get_learning_rate_scheduler(optimizer)


    train_loss_hist, valid_loss_hist = [], []

    for epoch in range(EPOCHS):
        cnt_train_loss = 0.0
        cnt_valid_loss = 0.0

        progress_bar = tqdm(total=num_train_batches)
        progress_bar.write(f"STARTING EPOCH {epoch+1}/{EPOCHS}")


        for index, (img, speed, target_vector, mask_vector) in enumerate(train_dataloader): #iterate training data 
            model.train()

            img = img / 255.
            predictions = model(img, speed)
            predictions = predictions * mask_vector

            loss = criterion(predictions, target_vector)
            cnt_train_loss += loss.item() / predictions.shape[0]

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

        with torch.no_grad(): # context manager for disabling autograd engine 
            model.eval()
            for index, (img, speed, target_vector, mask_vector) in enumerate(valid_dataloader): # iterate over validation batches
                predictions = model(img, speed)
                predictions = predictions * mask_vector
                
                loss = criterion(predictions, target_vector)
                cnt_valid_loss += loss.item() / predictions.shape[0]


        train_loss_hist.append( cnt_train_loss )
        valid_loss_hist.append( cnt_valid_loss )
        

        progress_bar.write(f"Train loss: {cnt_train_loss}\nValidation loss:{cnt_valid_loss}")

        if not (epoch % 5):
            progress_bar.write(f"Saving model for epoch {epoch+1}")
            # add saver

        if cnt_train_loss > max(valid_loss_hist[:-1]):
            progress_bar.write(f"Saving best model by validation")
            # add saver 

        progress_bar.close()
        progress_bar.write(EPOCH_SEPARATOR)

