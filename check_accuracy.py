import glob
import time

from keras.models import load_model
import sys, os
import keras.backend as K
from data_generator import MYGenerator

if __name__ == '__main__':

    base_dir ='./models/partially_trained_models/*/*'
    models = glob.glob(base_dir)
    for model_path in models:

        base_model = model_path.split('-')[0].split('\\')[-2]
        # if base_model in ['inception_resnet_v2', 'inception']:
        #     continue

        try:
            epoch = model_path.split('-')[0].split(' ')[0][-2:]
            loss = model_path.split('-')[1].split(' ')[1]
            acc = model_path.split('-')[2].split('acc')[1]

            model = load_model(model_path)
        except:
            print('could not open ' + model_path)
            continue

        batch_size = 32

        # get accuracy on validation
        validation_folder = r'./data/valid_0.1'
        labels_file = './data/valid_0.1_labels.csv'
        my_eval_batch_generator = MYGenerator(train_folder=validation_folder, labels_file=labels_file,
                                              batch_size=batch_size, preprocess_fun_name=base_model)

        num_classified_samples = len(os.listdir(validation_folder))
        t = time.time()
        val_loss, val_acc = model.evaluate_generator(generator=my_eval_batch_generator,
                                                     verbose=1,
                                                     steps=(num_classified_samples // batch_size),
                                                     use_multiprocessing=True,
                                                     workers=4,
                                                     max_queue_size=32)
        classification_time = (time.time() - t) / num_classified_samples

        # get accuracy on test
        test_folder = r'./data/test_0.1'
        labels_file = './data/test_0.1_labels.csv'
        my_eval_batch_generator = MYGenerator(train_folder=test_folder, labels_file=labels_file,
                                              batch_size=batch_size, preprocess_fun_name=base_model)

        num_classified_samples = len(os.listdir(test_folder))
        test_loss, test_acc = model.evaluate_generator(generator=my_eval_batch_generator,
                                             verbose=1,
                                             steps=(num_classified_samples // batch_size),
                                             use_multiprocessing=True,
                                             workers=4,
                                             max_queue_size=32)

        n_params = model.count_params()
        with open('stats.csv', 'a') as log:

            stats = f'{base_model},{epoch},{acc},{loss},{val_acc},{val_loss},{test_acc},{test_loss}' \
                f',{n_params},{classification_time},\n'
                      # .format(base_model=base_model,
                      #         acc=acc, loss=loss,
                      #         val_acc=val_acc, val_loss=val_loss,
                      #         ctime=classification_time,
                      #         epoch=epoch, n_params=n_params)
            print(stats)
            log.write(stats)

        del model
        K.clear_session()
