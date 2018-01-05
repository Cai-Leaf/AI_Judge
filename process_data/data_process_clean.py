from utils import file_pickle as fp

train = fp.get_pickle_data('../data_mid/train_data.pkl')
texts = train['word']
law = train['law']
panish_class = train['panish_class']


new_word = []
new_law = []
new_panish_class = []
for i in range(len(texts)):
    if len(texts[i]) >= 100:
        new_word.append(texts[i])
        new_law.append(law[i])
        new_panish_class.append(panish_class[i])

result = {
    'word': new_word,
    'panish_class': new_panish_class,
    'law': new_law
}
fp.save_pickle_data('../data_mid/train_data.pkl', result)
