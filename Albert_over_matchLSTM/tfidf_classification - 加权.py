# -*- coding: gbk -*-
from __future__ import print_function, unicode_literals
import sys
import smart_open
import jieba
import  pandas  as pd
import numpy as np
from gensim.models import Word2Vec,TfidfModel
from gensim import corpora
from keras.utils.np_utils import to_categorical
import os
import logging
from keras import models,layers,preprocessing,initializers
from keras.models import Sequential,Model,Input
from keras.layers import Flatten, Dense,Embedding,BatchNormalization,concatenate,GlobalAveragePooling1D,GlobalMaxPooling1D,AveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D,LSTM,SpatialDropout1D,GRU,Bidirectional
from keras.initializers import glorot_normal,orthogonal
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler
from keras import regularizers
import keras.backend as K


#read experiment data in excel
df=pd.read_excel('D:/Documents/agricultural_q_and_a/experimental_data1.xlsx')
question_data=[]
category_data=[]
for i in df.index.values:
    #read category and question
    question_row=df.loc[i,['question']].to_list()
    category_row=df.loc[i,['category_id']].to_list()
    question_data.append(question_row)
    category_data.append(category_row)
    
     
cat_id_df = df[['category_id', 'category_name']].drop_duplicates().sort_values('category_id').reset_index(drop=True) #read category, construct category_id, category_name row

#read sogou_dict and stop_dict
jieba.load_userdict("D:/Documents/agricultural_q_and_a/paper1/datasets/sougou_dict.txt")
stopwords = [line.strip() for line in open("D:/Documents/agricultural_q_and_a/paper1/datasets/stop_dict.txt", 'r', encoding='gbk').readlines()]  

#jieba word segmentation
cut_word_question=[]
for lines in question_data:
	outstr=[]
	for line in lines:
		words = jieba.cut(line.strip(),cut_all=True)
		for word in words:
			if word !=''  and word not in stopwords:
				outstr.append(word)				
#cut_word_question store all the segmented words		
	cut_word_question.append(outstr)

#calc TF-IDF for each word
dictionary = corpora.Dictionary(cut_word_question)
corpus = [dictionary.doc2bow(line) for line in cut_word_question]
tf_idf_model = TfidfModel(corpus, normalize=False)
word_tf_tdf = list(tf_idf_model[corpus])

tfidf_list = []
dic={}
for items in word_tf_tdf:
	tfidf_question = []
	for tuples in items:
		dic[dictionary[tuples[0]]]=tuples[1]
		tfidf_question.append(tuples[1])
	tfidf_list.append(tfidf_question)
print("Segmented sentence:\n",cut_word_question[:1])
print('The tf-idf value of each word in the question:\n',tfidf_list[:1])

index_list = []		
for line in tfidf_list:
	index_list.append(line.index(max(line)))



#count words
total_word=[]
for line in cut_word_question:	
	for word in line:														
		total_word.append(word) 
vocab = set(str(total_word))
less_word = []

#word2vec
MAX__LENGTH =128
question_model = Word2Vec(cut_word_question, sg=1,size=MAX__LENGTH, window=5, min_count=1, workers=2,iter=25)
question_model.save('./question_model')
question_model.wv.save_word2vec_format('./question_model.txt', binary=False)



embedding_dic={}
for k,v in question_model.wv.vocab.items():
	embedding_dic[k]=question_model.wv[k]

input_train=[]
for line in cut_word_question:#cut_word_question 
	input_row=[]
	for word in line:			
		if word in embedding_dic.keys():		
			value = embedding_dic[word]*dic[word]
			input_row.extend(value)
	connect_array = np.array(input_row).reshape(-1,MAX__LENGTH)
	input_train.append(connect_array.tolist())
	

num_list=[len(line) for line in input_train]
print("longest num_list is ",max(num_list))
print("longest num_list is ",max(num_list)) 

MAXLEN=100
#normalization of input sentences
x_train_all = pad_sequences(input_train,MAXLEN)
print('shape of x_train is :',x_train_all.shape)

#one-hot encode the category
one_hot_train_labels = to_categorical(category_data)
print('shape of one_hot_train_labels is :',one_hot_train_labels.shape)


#batch normalization
x_train, x_test,y_train,y_test = train_test_split(x_train_all,one_hot_train_labels , test_size = 0.10, random_state = 5)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

partial_x_train,x_val,partial_y_train ,y_val = train_test_split(x_train,y_train, test_size = 0.10, random_state = 6)
print(partial_x_train.shape,x_val.shape)
print(partial_y_train.shape,y_val.shape)

main_input = Input(shape=(MAXLEN,MAX__LENGTH))
bidirectional =Bidirectional(GRU(128,activation='relu',dropout=0.2,recurrent_dropout=0.2,return_sequences = True),merge_mode="concat")(main_input)
bidirectional=BatchNormalization()(bidirectional)
x = concatenate([bidirectional,main_input],axis=-1)

filter_lengths = [2,3,4,5,6]
conv_layers = []
conv_layers2 = []
for filter_length in filter_lengths:
	
	con_conv=[]
	conv_layer_1 = Conv1D(filters=64, kernel_size=filter_length, strides=1,
						  padding='same', activation='relu')(x)
	bn_layer1 = BatchNormalization()(conv_layer_1 )
	conv_layer_2 = Conv1D(filters=128, kernel_size=filter_length, strides=1,
						  padding='same', activation='relu')(x)
	bn_layer2 = BatchNormalization( )(conv_layer_2)
	con_conv = concatenate([bn_layer1,bn_layer2],axis=-1)
	conv_layers.append(con_conv)
poolings = [GlobalAveragePooling1D()(conv) for conv in conv_layers] + [GlobalMaxPooling1D()(conv) for conv in conv_layers]
cnn = concatenate(poolings)
dropout = Dropout(0.5)(cnn)
cnn = BatchNormalization()(cnn)
cnn=Dense(128,activation='relu')(dropout)
cnn=Dense(64,activation='relu')(cnn)
main_output = Dense(partial_y_train.shape[1], activation='softmax')(cnn)
model = Model(main_input, outputs = main_output)
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping,LearningRateScheduler

def scheduler(epoch):
	
	
        # epoch=10, learning rate decrease to 1/10
	if epoch % 10 == 0 and epoch != 0:
		lr = K.get_value(model.optimizer.lr)
		K.set_value(model.optimizer.lr, lr * 0.1)
		print("lr changed to {}".format(lr * 0.1))
	return K.get_value(model.optimizer.lr)
lr = LearningRateScheduler(scheduler)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3,factor=0.1,mode='auto')

filepath='weights.best.hdf5'
# the new max will overwrite the old max
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,save_best_only=True,mode='max',period=1)

earlyStopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
callbacks_list = [reduce_lr,checkpoint,earlyStopping,lr]#


history =model.fit(partial_x_train, partial_y_train, batch_size=256
, epochs=100,validation_data = (x_val, y_val),callbacks=callbacks_list)# 


#load best model
# load weights 加载模型权重 load model weight
#model.load_weights('weights.best.hdf5')  #wrong keras version might throw errors

# compile 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Created model and loaded weights from hdf5 file')

result = model.evaluate(x_test,y_test)
print("{0}: {1:.2f}%".format(model.metrics_names[1], result[1]*100))

#plot the train loss and validation loss
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plot accuracy of training and validation
plt.clf()     # clear previous plot
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.clf()


import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.font_manager import FontProperties
from  sklearn.metrics import classification_report
plt.rcParams['font.sans-serif'] = ['SimHei'] 
#sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']})
#myfont = FontProperties(fname=r"C:\Windows\Fonts\simfang.ttf",size=14)
y_pred = model.predict(x_test)
f.close()
y_pred = y_pred.argmax(axis = 1)
y_test = y_test.argmax(axis = 1)
print('accuracy %s' % accuracy_score(y_pred, y_test))
report = classification_report(y_test, y_pred,target_names=cat_id_df['category_name'].values,digits=4)
print(report)
with open('log_result.txt','w') as f:
    f.write(str(report))

#confusion matrix generation, f, annot=False, ax=ax
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,8))
#sns.set(font=myfont)

sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=cat_id_df['category_name'].values, yticklabels=cat_id_df['category_name'].values)
plt.ylabel('actual',fontsize=18)
plt.xlabel('prediction',fontsize=18)
plt.show()
