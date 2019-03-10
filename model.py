
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split

# ====================================================================
# model
# ====================================================================

MAX_TITLE_LEN = 30
VOCAB_SIZE = 100000
WORD_EMBEDDING_DIM = 256

FILTER_SIZES = [2, 3, 4]
NUM_FILTERS = 512
DROPOUT_RATE = 0.2

EPOCHS = 100
BATCH_SIZE = 30

inputs = Input(shape=(MAX_TITLE_LEN,))
embedding_layer = Dense(units=2000)(inputs) # Glove

# reshape = Reshape((MAX_TITLE_LEN, WORD_EMBEDDING_DIM, 1))(embedding_layer)
conv_0 = Conv2D(filters=NUM_FILTERS, kernel_size=(FILTER_SIZES[0], WORD_EMBEDDING_DIM, padding='valid', kernel_initializer='normal', activation='relu'))(embedding_layer)
maxpool_0 = MaxPool2D(pool_size=(MAX_TITLE_LEN - FILTER_SIZES[0] + 1, 1), strides=(1,1), padding='valid')(conv_0))
conv_1 = Conv2D(filters=NUM_FILTERS, kernel_size=(FILTER_SIZES[1], WORD_EMBEDDING_DIM, padding='valid', kernel_initializer='normal', activation='relu'))(maxpool_0)
maxpool_1 = MaxPool2D(pool_size=(MAX_TITLE_LEN - FILTER_SIZES[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
conv_2 = Conv2D(filters=NUM_FILTERS, kernel_size=(FILTER_SIZES[2], WORD_EMBEDDING_DIM, padding='valid', kernel_initializer='normal', activation='relu'))(maxpool_1)
maxpool_2 = MaxPool2D(pool_size=(MAX_TITLE_LEN - FILTER_SIZES[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

flatten = Flatten()(maxpool_2)
dropout = Dropout(rate=DROPOUT_RATE)(flatten)
outputs = Dense(units=10, activation="softmax")(dropout)

model = Model(inputs, outputs)
checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))
