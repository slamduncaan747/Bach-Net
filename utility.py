from music21 import *
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

part_names = ['soprano', 'alto', 'tenor', 'bass']

voice_ranges = {
    'soprano': {'min': 60, 'max': 79, 'size': 21},
    'alto': {'min': 55, 'max': 74, 'size': 21},
    'tenor': {'min': 50, 'max': 69, 'size': 21},
    'bass': {'min': 36, 'max': 62, 'size': 28}
}

def visualize_as_matrix(dataset_element, voice_ranges):
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))

    soprano = dataset_element['input'].reshape(21, 64)
    output = dataset_element['output']
    alto = output[0:1344].reshape(21, 64)
    tenor = output[1344:2688].reshape(21, 64)
    bass = output[2688:4480].reshape(28, 64)

    voices = [
        ('Soprano', soprano, voice_ranges['soprano']),
        ('Alto', alto, voice_ranges['alto']),
        ('Tenor', tenor, voice_ranges['tenor']),
        ('Bass', bass, voice_ranges['bass'])
    ]
    
    for idx, (name, data, vrange) in enumerate(voices):
        im = axes[idx].imshow(data, aspect='auto', origin='lower', cmap='Blues')
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Time Step (16th notes)')
        axes[idx].set_ylabel('Pitch Index (0={}, {}={})'.format(
            vrange['min'], vrange['size']-1, vrange['max']
        ))
        axes[idx].grid(True, alpha=0.3)
        for measure in range(1, 4):
            axes[idx].axvline(x=measure*16, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
def play_dataset_element(dataset_element, voice_ranges, tempo=60):
    score = stream.Score()
    score.insert(0, tempo_module.MetronomeMark(number=tempo))
    
    parts_data = [
        ('Soprano', dataset_element['input'].reshape(21, 64), voice_ranges['soprano']),
        ('Alto', dataset_element['output'][0:1344].reshape(21, 64), voice_ranges['alto']),
        ('Tenor', dataset_element['output'][1344:2688].reshape(21, 64), voice_ranges['tenor']),
        ('Bass', dataset_element['output'][2688:4480].reshape(28, 64), voice_ranges['bass'])
    ]
    
    for part_name, piano_roll, voice_range in parts_data:
        part = stream.Part()
        part.partName = part_name
        part.append(meter.TimeSignature('4/4'))
        
        i = 0
        while i < 64:
            pitch_indices = np.where(piano_roll[:-1, i] == 1)[0]
            
            if len(pitch_indices) > 0:
                pitch_index = pitch_indices[0]
                midi_pitch = pitch_index + voice_range['min']
                
                duration = 1
                j = i + 1
                while j < 64:
                    if piano_roll[-1, j] == 1:
                        duration += 1
                        j += 1
                    else:
                        break  
                
                n = note.Note(midi_pitch)
                n.quarterLength = duration * 0.25
                n.offset = i * 0.25
                part.append(n)
                
                i = j
            else:
                if piano_roll[-1, i] == 1:
                    i += 1
                else:
                    r = note.Rest(quarterLength=0.25)
                    r.offset = i * 0.25
                    part.append(r)
                    i += 1
        
        part.makeMeasures(inPlace=True)
        score.append(part)
    
    score.show('midi')
    return score



