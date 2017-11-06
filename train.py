"""
Train our RNN on bottlecap or prediction files generated from our CNN.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time

def train(data_type, seq_length, model, saved_model=None,
          concat=False, class_limit=None, image_shape=None,
          load_to_memory=False):
    # Set variables.
    nb_epoch = 1000
    batch_size = 8 #32 for LSTM; 16 for CRNN; 8 for CNN-3d

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath='/hdd/hpc/Projects/Weather/data_WJL/data/checkpoints/' + model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir='/hdd/hpc/Projects/Weather/data_WJL/data/logs')

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger('/hdd/hpc/Projects/Weather/data_WJL/data/logs/' + model + '-' + 'training-' + \
        str(timestamp) + '.log')

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    #steps_per_epoch = (len(data.data) * 0.7) // batch_size
    train, test, validation = data.split_train_test()
    steps_per_epoch = len(train) // batch_size

    if load_to_memory:
        # Get data.
        # X, y = data.get_all_sequences_in_memory(batch_size, 'train', data_type, concat)
        # X_val, y_val = data.get_all_sequences_in_memory(batch_size, 'validation', data_type, concat)
        # X_test, y_test = data.get_all_sequences_in_memory(batch_size, 'test', data_type, concat)
        X, y = data.get_all_sequences_in_memory('train', data_type, concat)
        X_val, y_val = data.get_all_sequences_in_memory('validation', data_type, concat)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type, concat)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type, concat)
        val_generator = data.frame_generator(batch_size, 'validation', data_type, concat)
        test_generator = data.frame_generator(batch_size, 'test', data_type, concat)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=[checkpointer, tb, early_stopper, csv_logger],
            # callbacks=[checkpointer, tb, csv_logger], #early_stopper, csv_logger],
            epochs=nb_epoch)

        # evaluate model
        results = rm.model.evaluate(
            X_test,
            y_test,
            batch_size=batch_size,
            verbose=1)
        print()
        print('Evaluation results on test data is:')
        print(results)
        print(rm.model.metrics_names)

    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[checkpointer, tb, early_stopper, csv_logger],
            # callbacks=[checkpointer, tb, csv_logger], #early_stopper, csv_logger],
            validation_data=val_generator,
            validation_steps=10)

        # Evaluate!
        results = rm.model.evaluate_generator(
            generator=test_generator,
            steps=len(test) // batch_size)
        print()
        print('Evaluation results on test data is:')
        print(results)
        print(rm.model.metrics_names)

def main():
    """These are the main training settings. Set each before running
    this file."""
    model = 'lrcn'  # see `models.py` for more
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 90
    load_to_memory = True  # pre-load the sequences into memory

    # Chose images or features and image shape based on network.
    if model == 'conv_3d' or model == 'crnn' or model == 'c3d':
        data_type = 'images'
        #image_shape = (80, 80, 3)
        image_shape = (144, 81, 3) # Work for crnn with batch size 8
        #image_shape = (112, 63, 3) # Work for 3dcnn with batch size 5
        load_to_memory = False
    elif model == 'lrcn':
        data_type = 'images'
        image_shape = (144, 81, 3)
    else:
        data_type = 'features'
        image_shape = None

    # MLP requires flattened features.
    if model == 'mlp':
        concat = True
    else:
        concat = False

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, concat=concat, image_shape=image_shape,
          load_to_memory=load_to_memory)

if __name__ == '__main__':
    main()
