import torch
import numpy as np
import pickle
from music21 import *
from utility import visualize_as_matrix, play_dataset_element, voice_ranges
from model import BachNet
from ChoraleDataset import ChoraleDataset
import os

def process_model_outputs(output_1, output_2, output_3, threshold=0.5):
    output_1 = torch.sigmoid(output_1)
    output_2 = torch.sigmoid(output_2)
    output_3 = torch.sigmoid(output_3)

    roll_1 = (output_1 > threshold).float().squeeze(0).squeeze(0).numpy()
    roll_2 = (output_2 > threshold).float().squeeze(0).squeeze(0).numpy()
    roll_3 = (output_3 > threshold).float().squeeze(0).squeeze(0).numpy()
    
    return [roll_1, roll_2, roll_3]

device = torch.device("cpu")
model = BachNet()
model.load_state_dict(torch.load('checkpoints/Copy of checkpoint200.pth', map_location=device)['model_state_dict'])
model.eval()

dataset = ChoraleDataset('dataset.pkl')
random_index = np.random.randint(0, len(dataset))
input_tensor, _, _, _ = dataset[random_index]
input_tensor = input_tensor.unsqueeze(0)

with torch.no_grad():
    output_1, output_2, output_3 = model(input_tensor)
print(output_1.shape, output_2.shape, output_3.shape)
print(output_1)
output_rolls = process_model_outputs(output_1, output_2, output_3)

original_sample = dataset.data[random_index]
dataset_element = {
    'chorale': original_sample['chorale'],
    'measures': original_sample['measures'],
    'input': original_sample['input'],
    'output': output_rolls
}

#visualize_as_matrix(dataset_element, voice_ranges)
midi = play_dataset_element(dataset_element, voice_ranges, bpm=100)
os.makedirs('samples', exist_ok=True)
midi.write('midi', fp='samples/chorale.mid')
