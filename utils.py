import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def load_data():
    '''
    load the data into a dict
    :return: dict of DataFrames ; dict
    '''
    path = '/Users/ronlitman/RonLitman/Work/Agoda/'
    dict_df = {}
    dict_df['p1'] = pd.read_csv(path + 'Partner1.csv')
    dict_df['p2'] = pd.read_csv(path + 'Partner2.csv')
    dict_df['label_sample'] = pd.read_csv(path + 'examples.csv')
    return dict_df


def print_stats(df, key):
    '''
    print general stats on the data
    :param df: DataFrame
    :param key: p1,p2 ; str
    '''
    print('\n')
    print('Stats for {}'.format(key))
    print('Number of unique hotel room are: {}'.format(df['{}.key'.format(key)].nunique()))
    print('Number of missing data per column:')
    print(df.isnull().sum(axis=0))
    print('unique number of values:')
    print(df.nunique())


def print_results_stats(pred, labels):
    '''
    print total correct pairs and the total amount that was labeled
    :param pred: prediction ; DataFrame
    :param labels: labels ; DataFrame
    '''
    correct = 0
    total_labeled = 0
    for pair1, pair2 in zip(pred['p1.key'], pred['p2.key']):
        if (pred[pred['p1.key'] == pair1]['score'] == -1).all():
            continue
        if (labels[labels['p1.key'] == pair1]['p2.key'] == pair2).all():
            correct = correct + 1
        total_labeled = total_labeled + 1
    print('Total correct {} from {} labeled ({}%)'.format(correct, total_labeled, correct / float(total_labeled)))
    print('Coverage: {}/{} ({}%)'.format(correct, len(labels), correct / float(len(labels))))
    plot_roc(pred, labels)


def plot_roc(pred,labels):
    all_acc = []
    all_coverage = []
    for i in range(100):
        temp = pred[pred.score > i]
        correct = 0
        total_labeled = 0
        for pair1, pair2 in zip(temp['p1.key'], temp['p2.key']):
            if (pred[pred['p1.key'] == pair1]['score'] == -1).all():
                continue
            if (labels[labels['p1.key'] == pair1]['p2.key'] == pair2).all():
                correct = correct + 1
            total_labeled = total_labeled + 1
        all_acc.append(correct / float(total_labeled))
        all_coverage.append(correct / float(len(labels)))

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(all_acc, all_coverage)
    plt.xlabel('Acc')
    plt.ylabel('Coverage')
    plt.title('Acc Vs Coverage given all thresholds')
    plt.savefig('roc.png', format="png")
    plt.show()


def plot_n_preds_per_thres(pred):
    n_pred = []
    for i in range(100):
        n_pred.append(len(pred[pred.score > i]))

    fig = plt.figure()
    ax = plt.axes()
    ax.plot([i for i in range(100)], n_pred)
    plt.xlabel('Threshold')
    plt.ylabel('# Pairs predicted')
    plt.title('Threshold Vs # of pairs predicted')
    plt.savefig('roc_pairs_pred.png', format="png")
    plt.show()