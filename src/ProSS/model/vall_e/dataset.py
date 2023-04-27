from collections import Counter
import re

def sentence2vec(sent):
    vocab = open('./word2idx.txt', 'r').read()
    vocab = vocab.split(' ')
    #print(vocab)
    vec = []
    for word in sent.split():
        vec.append(vocab.index(word))
    return vec


file = open('./tfotr_text.txt', 'r').read()
file = file.replace('\n', '')
file = re.split(r'[.?!]', file)
file2 = open('./tfotr_text_lines.txt', 'w')

for line in file:
    line2 = line.strip()
    line2 = line2.replace('\t', '')
    #line2 = line2.replace('?', ' ?\n')
    #line2 = line2.replace('.', ' .\n')
    #line2 = line2.replace('!', ' !\n')
    line2 = line2.replace(',', ' , ')
    line2 = line2.replace('_', '')
    line2 = line2.replace('-', ' - ')
    line2 = line2.replace('\'', ' \' ')
    line2 = line2.replace('\"', ' \" ')
    line2 = line2.replace('(', ' ( ')
    line2 = line2.replace(')', ' ) ')
    line2 = line2.replace(':', ' : ')
    line2 = line2.replace(';', ' ; ')
    line2 = line2.lower()
    
    print('line2: ', line2)
    if len(line2) > 10:
        line2 = line2.strip()
        file2.write(line2+'\n')

file2.close()


#newfile = open('./TFOTR.txt', 'w')
#newfile.write(file)

lines = open('./tfotr_text_lines.txt', 'r').readlines()
text5 = open('./tfotr_text_lines.txt', 'r').read()

text = []
for line in lines:
    line = line.strip()
    if len(line) > 1:
        if line[0].isalpha():
            text.append(line.strip())

print(text[0:30])
print(len(Counter(text5.split())))
all_words = sorted(Counter(text5.split()).keys())

file2 = open('./word2idx.txt', 'w')

for word in all_words:
    file2.write(word + ' ')

file2.write('<PAD>' + ' ')
file2.write('<EOS>' + ' ')
file2.write('<BOS>' + ' ')

#print(all_words)
#newfile.close()
file2.close()

newfile = open('./TFOTR.txt', 'w')
newfile.writelines(text)
newfile.close()

print(sentence2vec("how are you "))