import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from model import YOLOv1
from dataset import VOCDataset
from loss import YOLOv1Loss
from utils import IoU, NMS, mAP, save_checkpoint, load_checkpoint


seed = 42
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 20
NUM_WORKERS = 2
PIN_MEMORY = True
ROOT_DIR = "data"
LOAD_MODEL = False
LOAD_MODEL_FILE = ""

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, label):
        for tsfm in self.transforms:
            image = tsfm(image)
            
        return image, label


transform = Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]
)


def train_one_epoch(train_loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for _, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        loss = loss_fn(output, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    

def main():
    # ============ #
    #     Model    #
    # ============ #
    model = YOLOv1()
    model = model.to(DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YOLOv1Loss()
    
    if LOAD_MODEL:
        load_checkpoint(
            checkpoint=torch.load(LOAD_MODEL_FILE),
            model=model,
            optimizer=optimizer
        )
        
    # ============ #
    #    Dataset   #
    # ============ #
    train_dataset = VOCDataset(
        csv_file="data/100examples.csv",
        root_dir=ROOT_DIR,
        transform=transform
    )
    
    test_dataset = VOCDataset(
        csv_file="data/test.csv",
        root_dir=ROOT_DIR,
        transform=transform
    )
    
    # ============ #
    #  DataLoader  #
    # ============ #
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    # ============ #
    #   Training   #
    # ============ #
    for epoch in range(EPOCHS):
        print(f"==>> Epoch {epoch}")
        train_one_epoch(train_loader, model, optimizer, loss_fn)
        

if __name__ == "__main__":
    main()
    