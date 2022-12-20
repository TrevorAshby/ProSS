from tacotron2.text import text_to_sequence
import numpy as np
import torch

# tones: 'H':0, 'L':1, '+':2, '*':3, '-':4, '!':5, '%':6, 'NULL':-1
TONES = {
    # -2 is what indicates that these markings begin
    'N':-1,
    'H':0,
    'L':1,
    '+':2,
    '*':3,
    '-':4, 
    '!':5, 
    '%':6,
    '1':7,
    '2':8,
    '3':9,
    '4':10
}

def encode_text_with_tobi(aligned_anno):
    #print([tone_encoding(x[1], x[2]) for x in aligned_anno if x[0] != ''])
    z = [-2]
    for x in aligned_anno:
        #print(x)
        if x[0] != '':
            te = tone_encoding(x[1], x[2])
            for e in te:
                z.append(e)
    #print(z)
    seq = np.append(np.array(text_to_sequence(''.join([x[0] + ' ' for x in aligned_anno])+'.', ['english_cleaners'])), z)
    #print(seq)
    seq = torch.autograd.Variable(
        torch.from_numpy(seq)).long()
    #print(seq)
    return seq

def tone_encoding(tone, brk):
    arr = []
    for char in tone:
        arr.append(TONES[char])
    arr.append(TONES[brk])
    return arr

def align_annotation(words, tones, breaks):
    #w_idx = 0
    a_idx = 0
    a_len = len(tones)
    b_idx = 0
    b_len = len(breaks)
    final_array = []
    # TONES
    #while 1:
    #print(words)
    #print()
    tones = sorted(tones, key=lambda x: x[1])
    #print(tones)
    #print()
    #print(breaks)
    for w_idx in range(len(words)):
        if a_idx < a_len:
            anno = tones[a_idx]
        else:
            anno = None
        
        if b_idx < b_len:
            brk = breaks[b_idx]
        else:
            brk = None
        
        wrd = words[w_idx]

        # if anno x larger, place null, move to next word
        fin = [wrd.text]
        if anno == None:
            fin.append('N')
        elif wrd.xmax < anno.xpos:
            fin.append('N')

        # check if annotation fits in window
        elif wrd.xmax >= anno.xpos and wrd.xmin <= anno.xpos:
            fin.append(anno.text)
            a_idx += 1
        else:
            fin.append('N')

        if brk == None:
            fin.append('N')
        elif wrd.xmax < brk.xpos:
            fin.append('N')
        elif wrd.xmax >= brk.xpos and wrd.xmin <= brk.xpos:
            fin.append(brk.text)
            b_idx += 1

        final_array.append(fin)
    return final_array