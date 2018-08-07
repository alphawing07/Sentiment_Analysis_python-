# Sentiment classification using Deep neural networks

Sentiment analysis involves extraction of subjective information from documents
like online reviews to determine the polarity with respect to certain objects. It is useful
for identifying trends of public opinion in the social media, for the purpose of
marketing and consumer research.

## Dataset information
The training dataset is csv file containing 1.6 million tweets and statements without emoticons marked with polarities 
4(negetive) ,0 (positive) which are converted to [0,1] (for negetive),[1,0] (for positive)
,similarly testing dataset is also a csv file containing few statements, these 
files are preprocessed before training the neural network and converted from ```polarity,id,Date,query,user,tweet``` to the type ```sentiment,tweet```  using preprocess.py

## Requirements
library requirements for the project are 
 * nltk
 * pickle 
 * pandas
 * numpy 
 * tensorflow 
 ## Import Modules
 ```python
  import nltk
  from nltk.tokenize import word_tokenize
  from nltk.stem import WordNetLemmatizer
  import pickle
  import numpy as np 
  import pandas as pd
  import tensorflow as tf
 ```
 ## Pipeline 
 ### 1.Preprocessing 
 1. Run preprocess.py
    
    The above step will create a lexicon,shuffle the test data, covert testing and training data into desired format. The       training and test data files will be saved as 'train_set_shuffled.csv' and 'processed-test-set.csv'
 ```python
 def init_process(fin,fout):
	outfile = open(fout,'a')
	with open(fin, buffering=200000, encoding='latin-1') as f:
		try:
			for line in f:
				line = line.replace('"','')
				initial_polarity = line.split(',')[0]
				if initial_polarity == '0':
					initial_polarity = [1,0]
				elif initial_polarity == '4':
					initial_polarity = [0,1]

				tweet = line.split(',')[-1]
				outline = str(initial_polarity)+':::'+tweet
				outfile.write(outline)
		except Exception as e:
			print(str(e))
	outfile.close()

init_process('training.1600000.processed.noemoticon.csv','train_set.csv')
init_process('testdata.manual.2009.06.14.csv','test_set.csv')


def create_lexicon(fin):
	lexicon = []
	with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
		try:
			counter = 1
			content = ''
			for line in f:
				counter += 1
				if (counter/2500.0).is_integer():
					tweet = line.split(':::')[1]
					content += ' '+tweet
					words = word_tokenize(content)
					words = [lemmatizer.lemmatize(i) for i in words]
					lexicon = list(set(lexicon + words))
					print(counter, len(lexicon))

		except Exception as e:
			print(str(e))

	with open('lexicon-2500-2638.pickle','wb') as f:
		pickle.dump(lexicon,f)

create_lexicon('train_set.csv')


def convert_to_vec(fin,fout,lexicon_pickle):
	with open(lexicon_pickle,'rb') as f:
		lexicon = pickle.load(f)
	outfile = open(fout,'a')
	with open(fin, buffering=20000, encoding='latin-1') as f:
		counter = 0
		for line in f:
			counter +=1
			label = line.split(':::')[0]
			tweet = line.split(':::')[1]
			current_words = word_tokenize(tweet.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]

			features = np.zeros(len(lexicon))

			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					# OR DO +=1, test both
					features[index_value] += 1

			features = list(features)
			outline = str(features)+'::'+str(label)+'\n'
			outfile.write(outline)

		print(counter)

convert_to_vec('test_set.csv','processed-test-set.csv','lexicon-2500-2638.pickle')


def shuffle_data(fin):
	df = pd.read_csv(fin, error_bad_lines=False)
	df = df.iloc[np.random.permutation(len(df))]
	print(df.head())
	df.to_csv('train_set_shuffled.csv', index=False)
	
shuffle_data('train_set.csv')


def create_test_data_pickle(fin):

	feature_sets = []
	labels = []
	counter = 0
	with open(fin, buffering=20000) as f:
		for line in f:
			try:
				features = list(eval(line.split('::')[0]))
				label = list(eval(line.split('::')[1]))

				feature_sets.append(features)
				labels.append(label)
				counter += 1
			except:
				pass
	print(counter)
	feature_sets = np.array(feature_sets)
	labels = np.array(labels)

create_test_data_pickle('processed-test-set.csv')

```

### 2. Training 
Run neuralnetwork.py. This will train the 3 layers of neural network model on training data and 
test it against the testing data returning the accuracy achieved.
```python
lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 2

batch_size = 32
total_batches = int(1600000/batch_size)
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']
    return output

saver = tf.train.Saver()
tf_log = 'tf.log'

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
            print('STARTING:',epoch)
        except:
            epoch = 1

        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess,"./model.ckpt")
            epoch_loss = 1
            with open('lexicon.pickle','rb') as f:
                lexicon = pickle.load(f)
                print('lexicon length:',len(lexicon))
            with open('train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))

                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            # OR DO +=1, test both
                            features[index_value] += 1
                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x),
                                                                  y: np.array(batch_y)})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run +=1
                        print('Batch run:',batches_run,'/',total_batches,'| Epoch:',epoch,'| Batch Loss:',c,)

            saver.save(sess, "./model.ckpt")
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n') 
            epoch +=1

train_neural_network(x)

def test_neural_network():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            try:
                saver.restore(sess,"./model.ckpt")
            except Exception as e:
                print(str(e))
            epoch_loss = 0
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        feature_sets = []
        labels = []
        counter = 0
        with open('processed-test-set.csv', buffering=20000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    counter += 1
                except:
                    pass
        print('Tested',counter,'samples.')
        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))


test_neural_network()
```
## 3.Testing on manual statements 
test.py is a script which uses the above network after training . All it does is take a string input, vectorize it according to our bag of words model, feed it through the neural network, and the output will either be [1,0] for positive sentiment or [0,1] for negative sentiment.
```python
lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 2
hm_data = 2000000

batch_size = 32
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')


current_epoch = tf.Variable(1)

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']

    return output

saver = tf.train.import_meta_graph('./model.ckpt.meta')

def use_neural_network(input_data):
    prediction = neural_network_model(x)
    with open('lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)
        
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,"model.ckpt")
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
                features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        if result[0] == 0:
            print('Positive:',input_data)
        elif result[0] == 1:
            print('Negative:',input_data)

use_neural_network("It was the worst product.")
use_neural_network("Good job.")
```
#### output for test.py
![](https://user-images.githubusercontent.com/31281151/43791805-7f362868-9a94-11e8-9f36-fd3f81219900.png)

## Informaion about other files 
```training.1600000.processed.noemoticon.csv``` - Raw training data set
```testdata.manual.2009.06.14.csv```            - Testing data set
