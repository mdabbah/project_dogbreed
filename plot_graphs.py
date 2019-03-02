import matplotlib.pylab as plt
import pandas as pd
import numpy as np


def normalize(data: list):
    """
    returns the same list but normalized by it's sum
    :param data: list of floats
    :return: normalized data
    """
    return np.array(data) / np.sum(data)


def auto_label(rects, labels):
    """
    Attach a text label above each bar displaying its height
    """
    for rec_idx, rect in enumerate(rects):
        height = rect.get_height()
        label = str(labels[rec_idx])
        plt.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                 label,
                 ha='center', va='bottom')


if __name__ == '__main__':

    data_table = pd.read_csv('./stats.csv')
    num_classified_samples = 1022  # on validation for time normalization
    base_models = np.sort(list(set(data_table.base_model)))
    num_params = [data_table.loc[data_table.base_model == base_model, 'num_params'].iloc[0]*1e-3 for base_model
                  in base_models]
    classification_time = [np.mean(data_table.loc[data_table.base_model == base_model, 'classification_time']) for base_model
                           in base_models]
    classification_time = np.round(classification_time, 4)

    # plot accuracy vs epoch for all models
    training_acc_fig = plt.figure()
    training_loss_fig = plt.figure()
    validation_acc_fig = plt.figure()
    validation_loss_fig = plt.figure()

    figs = {training_acc_fig: 'accuracy on training', validation_acc_fig: 'accuracy on validation',
            training_loss_fig: 'loss on training', validation_loss_fig: 'loss on validation'}
    test_accuracy_per_model = []
    best_epoch_per_model = []
    for base_model in base_models:
        train_accuracy = data_table.loc[data_table.base_model == base_model, 'train_accuracy'].tolist()
        plt.figure(training_acc_fig.number)
        plt.plot(train_accuracy)

        train_loss = data_table.loc[data_table.base_model == base_model, 'train_loss'].tolist()
        plt.figure(training_loss_fig.number)
        plt.plot(train_loss)

        valid_loss = data_table.loc[data_table.base_model == base_model, 'val_loss'].tolist()
        plt.figure(validation_loss_fig.number)
        plt.plot(valid_loss)

        valid_accuracy = data_table.loc[data_table.base_model == base_model, 'val_accuracy'].tolist()
        test_accuracy = data_table.loc[data_table.base_model == base_model, 'test_accuracy'].tolist()
        plt.figure(validation_acc_fig.number)
        plt.plot(valid_accuracy)

        max_val_accuracy_idx = np.argmax(valid_accuracy)
        test_accuracy_per_model.append(np.round(test_accuracy[max_val_accuracy_idx]*100, 2))
        best_epoch_per_model.append(max_val_accuracy_idx)

    for fig, title in figs.items():
        plt.figure(fig.number)
        plt.xlabel('# epochs')
        plt.ylabel('acc %' if 'acc' in title else 'loss')
        plt.legend(base_models)
        plt.show(block=False)
        plt.title(title)
        plt.grid(True)

    width = 0.2
    num_params_fig = plt.figure()
    x = np.arange(len(num_params))
    rec1 = plt.bar(x, normalize(num_params), width)
    rec2 = plt.bar(x+width, normalize(classification_time), width)
    rec3 = plt.bar(x+2*width, normalize(test_accuracy_per_model), width)

    auto_label(rec1, num_params)
    auto_label(rec2, classification_time)
    auto_label(rec3, test_accuracy_per_model)

    plt.legend(['number of parameters [*1e3]', 'classification time per image [Sec]', 'accuracy on test set'])

    x_ticks = [model_name + ' #'+ str(epoch) for model_name, epoch in zip(base_models, best_epoch_per_model)]
    plt.xticks(x+width, x_ticks)
    plt.yticks([])
    plt.title('number of parameters per model')
    plt.show()
    var = 5

