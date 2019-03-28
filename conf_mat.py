import glob
import time
import numpy as np, pandas as pd
from keras.models import load_model
import sys, os
import keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix
import plotly.plotly as py
from plotly.offline import plot
import plotly.graph_objs as go

from data_generator import MYGenerator
from keras.metrics import top_k_categorical_accuracy


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


if __name__ == '__main__':

    base_dir ='./best_models/inception_resnet_v2 -epoch 15 -loss 1.739 -acc0.5565 -val_loss  1.509 -val_acc 0.7101*'
    models = glob.glob(base_dir)
    for model_path in models:

        base_model = model_path.split('-')[0].split('\\')[1].strip()
        # if base_model in ['inception_resnet_v2', 'inception']:
        #     continue

        try:
            epoch = model_path.split('-epoch')[1].split(' ')[1]
            loss = model_path.split('-acc')[1].split(' ')[1]
            acc = model_path.split('-loss')[1].split(' ')[1]

            model = load_model(model_path)
            # model.compile(optimizer=model.optimizer,
            #               loss=model.loss,
            #               metrics=model.metrics + [top_3_accuracy])
        except:
            print('could not open ' + model_path)
            continue

        batch_size = 32
        # get accuracy on test
        test_folder = r'./data/test_0.1'
        labels_file = './data/test_0.1_labels.csv'
        my_eval_batch_generator = MYGenerator(train_folder=test_folder, labels_file=labels_file,
                                              batch_size=batch_size, preprocess_fun_name=base_model.split('_aug')[0])

        num_classified_samples = len(os.listdir(test_folder))

        Y_pred = model.predict_generator(my_eval_batch_generator, num_classified_samples // batch_size + 1)
        y_pred = np.argmax(Y_pred, axis=1)
        print(f'Confusion Matrix for {base_model}')
        conf_mat = confusion_matrix(my_eval_batch_generator.labels, y_pred)
        target_names = pd.read_csv('./data/classes.csv')['class'].tolist()
        conf_mat_no_diag = np.array(conf_mat)
        conf_mat_no_diag[np.diag_indices(len(target_names))] = 0
        trace = go.Heatmap(z=conf_mat,
                           x=target_names,
                           y=target_names)

        most_confusing_class_ind = np.unravel_index(conf_mat.argmax(), conf_mat.shape)
        gt_most_confused = target_names[most_confusing_class_ind[0]]
        pred_most_confused = target_names[most_confusing_class_ind[1]]
        print(f'in model {base_model} most confusing classes are: class {gt_most_confused} (gt) is confused for {pred_most_confused} (pred) ')
        data = [trace]
        layout = go.Layout(title=f'confusion_matrix {base_model}')
        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename=f'confusion_matrix{base_model}.html')

        conf_mat_no_diag[np.diag_indices(len(target_names))] = 0

        print(conf_mat)
        print(f'Classification Report for {base_model}')
        print(classification_report(my_eval_batch_generator.labels, y_pred, target_names=target_names))



        del model
        K.clear_session()
