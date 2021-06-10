from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.metrics import categorical_crossentropy, categorical_accuracy

def build_NN(train,test,val,train_label,test_label,val_label,hidden_layers=5,hidden_units=[9,9,6,6,6],
	activations=['relu','relu','relu','relu','relu'],optimizer='adam',loss='categorical_crossentropy',
	metrics=['categorical_accuracy'],epochs=20,batch_size=20,graphs=False):
	# create a deep neural network model
	num_features = train.shape[1]
	num_output_classes = train_label.shape[1]
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
	dnn.compile(loss=loss, optimizer=optimizer,metrics=metrics)
	history = dnn.fit(train train_label, epochs=epochs, batch_size=batch_size,validation_data=(val, val_label))

	if graphs:
		# plot model loss while training
		epochs_arr = np.arange(1, epochs + 1, 1)
		my_history = history.history
		line1 = plt.plot(epochs_arr, my_history['loss'], 'r-', label='training loss')
		line2 = plt.plot(epochs_arr, my_history['val_loss'], 'b-', label='validation loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.title('Model loss')
		plt.legend()
		plt.show()

		# plot model accuracy while training
		line1 = plt.plot(epochs_arr, my_history['categorical_accuracy'], 'r-', label='training accuracy')
		line2 = plt.plot(epochs_arr, my_history['val_categorical_accuracy'], 'b-', label='validation accuracy')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.title('Model accuracy')
		plt.legend()
		plt.show()

		preds = pd.DataFrame(dnn.predict(X_validation))
		preds = preds.idxmax(axis=1)
		y_validation = y_validation.dot([0,1,2])
		model_acc = (preds == y_validation).sum().astype(float) / len(preds) * 100

		print('Deep Neural Network')
		print('Validation Accuracy: %3.5f' % (model_acc))

		preds_test = pd.DataFrame(dnn.predict(X_test))
		preds_test = preds_test.idxmax(axis=1)
		y_test = y_test.dot([0,1,2])
		model_acc = (preds_test == y_test).sum().astype(float) / len(preds_test) * 100
		print('Deep Neural Network')
		print('Test Accuracy: %3.5f' % (model_acc))
		# plot confusion matrix
		labels = np.unique(sdss_df['class'])
		ax = plt.subplot(1, 1, 1)
		ax.set_aspect(1)
		plt.subplots_adjust(wspace = 0.3)
		sns.heatmap(confusion_matrix(y_test, preds_test), annot=True,fmt='d', 
			xticklabels = labels, yticklabels = labels,cbar_kws={'orientation': 'horizontal'})
		plt.xlabel('Actual values')
		plt.ylabel('Predicted values')
		plt.title('Deep Neural Network')
		plt.show()
