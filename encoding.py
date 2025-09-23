from music21 import *
from music21 import analysis
import numpy as np
from utility import visualize_as_matrix, play_dataset_element, voice_ranges, part_names
import pickle

# Get all bach chorales and ensure they have exactly 4 parts and are in 4/4 time
songs = corpus.getComposer('bach')
valid_songs = []
for song in songs:
    chorale = converter.parse(song)
    ts = chorale.parts[0].getElementsByClass('Measure')[0].getTimeSignatures()
    if ts and ts[0].ratioString != '4/4':
        continue
    if len(chorale.parts) == 4:
        valid_songs.append(song)
    

def process_part(part, start_measure, voice_range):
    # For a given part and start measure, return piano roll representation
    # of size (21, 64), or (28, 64) depending on part
    roll = np.zeros((voice_range['size'], 64))

    # Ensure validity
    try:
        segment = part.measures(start_measure, start_measure + 3)
    except:
        return None 
    
    if len(segment.flatten().notes) == 0:
        return None 
    
    # Offset info
    first = segment.getElementsByClass('Measure')[0]
    segmentOffset = first.offset
    
    for element in segment.flatten().notes:
        if not isinstance(element, (note.Note, chord.Chord)):
            continue
        # Get the start and end positions for the note in 16th note quantized time
        relative_offset = element.offset - segmentOffset
        fixedPos = int(round(relative_offset * 4))
        fixedDuration = int(round(element.quarterLength * 4))
        endPos = fixedPos + fixedDuration

        # Safety checks
        if fixedPos < 0:
            fixedPos = 0
        if fixedPos >= 64:
            continue 
        if endPos > 64:
            endPos = 64  
        
        # Get pitch
        if isinstance(element, note.Note):
            pitch = element.pitch.midi
        elif isinstance(element, chord.Chord):
            pitch = max(p.midi for p in element.pitches)
        else:
            continue
        
        if pitch < voice_range['min'] or pitch > voice_range['max']:
            return None  
        
        # Set piano roll one-hot values, as well as hold tokens throughout duration of note
        pitch_index = pitch - voice_range['min']
        roll[pitch_index, fixedPos] = 1
        hold_index = voice_range['size'] - 1
        if fixedPos + 1 < 64:
            roll[hold_index, fixedPos + 1:min(endPos, 64)] = 1
    
    return roll

def get_buffers(chorale, start_measure):
    minBuffers = []
    maxBuffers = []
    segment = chorale.measures(start_measure, start_measure + 3)
    for i in range(4):
        part = segment.parts[i]
        for element in part.flatten().notes:
            if not isinstance(element, (note.Note, chord.Chord)):
                continue
            midis = [n.pitch.midi for n in part.flatten().notes if isinstance(n, note.Note)]
            chords = [c for c in part.flatten().notes if isinstance(c, chord.Chord)]
            if len(chords) > 0:
                midis.append(max(ch.pitches[0].midi for ch in chords))
            low = min(midis)
            high = max(midis)
            minBuffers.append(low - voice_ranges[part_names[i]]['min'])
            maxBuffers.append(voice_ranges[part_names[i]]['max'] - high)
    if(minBuffers == [] or maxBuffers == []):
        return []
    minBuffer, maxBuffer = min(minBuffers), max(maxBuffers)
    transpositions = [n for n in range(-minBuffer, maxBuffer) if n != 0]
    return transpositions

# For all valid songs, extract all valid 4 measure segments, proces into piano roll
# separate into inputs and outputs, and save to dataset file.
dataset = []
for filename in valid_songs:
    chorale = converter.parse(filename)
    measure_count = len(chorale.parts[0].getElementsByClass('Measure'))
    
    for start in range(1, measure_count - 3):
        x = None
        y = None

        invalid = False

        transpositions = get_buffers(chorale, start)
        #print(transpositions)
        for semitones in transpositions:
            # Loop through each part, process it, and add it to the dataset
            output = []
            for i in range(4):
                processed_part = process_part(chorale.parts[i].transpose(semitones), start, voice_ranges[part_names[i]])
                
                if processed_part is not None:
                    if i == 0:
                        x = processed_part
                    else:
                        output.append(processed_part)
                else:
                    invalid = True
                    break

            if invalid:
                continue   

            dataset.append({
                'chorale': filename,
                'transpose': semitones,
                'measures': (start, start+3),
                'input': x,
                'output': output
            })

print("Size: ", len(dataset))
with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
