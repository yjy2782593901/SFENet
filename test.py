import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dataprocessing import RoadDataset
from models.sfenet import SFENet
from models.losses import bic_iou, DiceLoss
from eval.evaluate import evaluate_online

to_pil = transforms.ToPILImage()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
imgsz = 512
batchsz = 1

ImgPath = "datasets/CrackMap/test_img"
GTPath = "datasets/CrackMap/test_lab"

TestingSet = RoadDataset(ImgPath, GTPath, imgsz, imgsz)
TestingLoader = DataLoader(TestingSet, batch_size=batchsz, shuffle=False)

Network = SFENet(embed_dim=[16, 32, 64, 128, 256], depth=[3, 3, 3, 3, 3])
checkpoint = torch.load('checkpoint/CrackMap.pth', map_location=DEVICE)
Network.load_state_dict(checkpoint['model'])

if torch.cuda.is_available():
    Network = Network.to(DEVICE)

Network.eval()

final_test_metrics = evaluate_online(
    model=Network,
    data_loader=TestingLoader,
    device=DEVICE,
    epoch=0,
    save_path='out/test_results/',
)

dummy_input = torch.randn(1, 3, imgsz, imgsz).to(DEVICE)
Network.eval()
with torch.no_grad():
    for _ in range(10):
        _ = Network(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    num_runs = 100
    start = time.time()
    for _ in range(num_runs):
        _ = Network(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    fps = num_runs / (end - start)
    final_test_metrics['FPS'] = fps

print("Final model performance on test set:")
for key, value in final_test_metrics.items():
    if isinstance(value, float):
        print(f'Test {key} -> {value:.4f}')

