from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.metrics import categorical_crossentropy, categorical_accuracy

def build_NN(train,test,val,train_label,test_label,val_label,hidden_layers=5,hidden_units=[9,9,6,6,6],activations=['relu','relu','relu','relu','relu'],
	optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'],epochs=20,batch_size=20,graphs=False):
	# create a deep neural network model
	num_features = train.shape[1]
	num_output_classes = test_label.shape[1]
	dnn = Sequential()
	dnn.add(Dense(9, input_dim=num_features, activation='relu'))
	dnn.add(Dropout(0.1))
	for i in range(1,hidden_layers):
		dnn.add(Dense(hidden_units[i], activation=activations[i]))
		ans = input("Do you want to add Dropout?(Y/N)")
		if ans=='Y':
			dropout_val = input("Please input dropout value")
			dnn.add(Dropout(dropout_val))
	dnn.add(Dense(num_output_classes, activation='softmax', name='output'))
	dnn.compile(loss=loss, optimizer=optimizer, 
              metrics=metrics)
	history = dnn.fit(train train_label, epochs=epochs, batch_size=batch_size,
                    validation_data=(val, val_label))
