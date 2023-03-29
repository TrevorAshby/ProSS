from collections import Counter

def sentence2vec(sent):
    vocab = open('./word2idx.txt', 'r').read()
    vocab = vocab.split(' ')
    #print(vocab)
    vec = []
    for word in sent.split():
        vec.append(vocab.index(word))
    return vec


file = open('./01 - The Fellowship Of The Ring.txt', 'r').read()
file = file.replace('\t', '')
file = file.replace('\n', '')
file = file.replace(',', ' , ')
file = file.replace('-', ' - ')
file = file.replace('\'', ' \' ')
file = file.replace('\"', ' \" ')
file = file.replace('(', ' ( ')
file = file.replace(')', ' ) ')
file = file.replace(':', ' : ')
file = file.replace(';', ' ; ')
file = file.replace('?', ' ?\n')
file = file.replace('.', ' .\n')
file = file.replace('!', ' !\n')
file = file.lower()

newfile = open('./TFOTR.txt', 'w')
newfile.write(file)

lines = open('./TFOTR.txt', 'r').readlines()
text5 = open('./TFOTR.txt', 'r').read()

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
newfile.close()
file2.close()

newfile = open('./TFOTR.txt', 'w')
newfile.writelines(text)
newfile.close()

print(sentence2vec("how are you ?"))