!wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip -P drive/hs-classification

import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

df = pd.read_csv('labeler1.txt', sep='\t', lineterminator='\r')
df.rename(columns={"1939": "query"})
df['labels'] = df[df.columns[1:]].apply(lambda x: '|'.join(x.dropna().astype(str)),axis=1)
df = df[['1939','labels']]
cleaned = df.set_index('1939').labels.str.split('|', expand=True).stack()

X = pd.DataFrame(df.iloc[:,0].values)
y = pd.DataFrame(df.iloc[:,-1].values)  

MAX_SEQUENCE_LENGTH = 20
MAX_WORDS = 2000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


# data cleaning
def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  string = re.sub('\d','', string)
  return string.strip().lower()


df['1939'] = df['1939'].apply(lambda x: clean_str(str(x)))

#create the word index dictionary
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(list(df['1939']))
sequences = tokenizer.texts_to_sequences(df['1939'])
    
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer(sparse_output=True)
labels = lb.fit_transform(df['labels'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    
#loading the pretrained embeddings
embeddings_index = {}

f = open("glove.6B.100d.txt",encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
    
def loadEmbeddingMatrix(typeToLoad):
        #load different embedding file from Kaggle depending on which embedding 
        #matrix we are going to experiment with
        if(typeToLoad=="glove"):
            EMBEDDING_FILE='drive/hs-classification/glove.6B.100d.txt'
            embed_size = 25
#         elif(typeToLoad=="word2vec"):
#             word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)
#             embed_size = 300
        elif(typeToLoad=="fasttext"):
            EMBEDDING_FILE='drive/hs-classification/wiki.simple.vec'
            embed_size = 300

        if(typeToLoad=="glove" or typeToLoad=="fasttext" ):
            embeddings_index = dict()
            #Transfer the embedding weights into a dictionary by iterating through every line of the file.
            f = open(EMBEDDING_FILE)
            for line in f:
                #split up line into an indexed array
                values = line.split()
                #first index is word
                word = values[0]
                #store the rest of the values in the array as a new array
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs #50 dimensions
            f.close()
            print('Loaded %s word vectors.' % len(embeddings_index))
        else:
            embeddings_index = dict()
            for word in word2vecDict.wv.vocab:
                embeddings_index[word] = word2vecDict.word_vec(word)
            print('Loaded %s word vectors.' % len(embeddings_index))
            
        gc.collect()
        #We get the mean and standard deviation of the embedding weights so that we could maintain the 
        #same statistics for the rest of our own random generated weights. 
        all_embs = np.stack(list(embeddings_index.values()))
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        
        nb_words = len(tokenizer.word_index)
        #We are going to set the embedding size to the pretrained dimension as we are replicating it.
        #the size will be Number of Words in Vocab X Embedding Size
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        gc.collect()

        #With the newly created embedding matrix, we'll fill it up with the words that we have in both 
        #our own dictionary and loaded pretrained embedding. 
        embeddedCount = 0
        for word, i in tokenizer.word_index.items():
            i-=1
            #then we see if this word is in glove's dictionary, if yes, get the corresponding weights
            embedding_vector = embeddings_index.get(word)
            #and store inside the embedding matrix that we will train later on.
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
                embeddedCount+=1
        print('total embedded:',embeddedCount,'common words')
        
        del(embeddings_index)
        gc.collect()
        
        #finally, return the embedding matrix
        return embedding_matrix
    
# model 1
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
model.add(Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1)))
model.add(Conv1D(64, kernel_size=10, padding='same', activation='relu'))
#google how to choose filter sizes
model.add(Conv1D(64, kernel_size=15, padding='same', activation='selu'))
model.add(Conv1D(128, kernel_size=15, padding='same', activation='relu'))
model.add(Conv1D(64, kernel_size=25, padding='same', activation='softmax'))
model.add(Conv1D(128, kernel_size=15, padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(688, activation='softmax'))

#TODO: Use sgd?
sgd = optimizers.SGD(lr=0.01, decay=1e-6)
model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=["accuracy"])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=30, batch_size=256)
