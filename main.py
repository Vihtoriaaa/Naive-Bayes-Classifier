import pandas as pd
from bayesian_classifier import BayesianClassifier, STOP_WORDS


def process_data(data_file):
    """
    Function for data processing and split it into X and y sets.
    :param data_file: str - train datado a research of your own
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """
    df = pd.read_csv(data_file)
    df['Message'] = df['Message'].str.lower()
    df['Message'] = df['Message'].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in STOP_WORDS]))
    df["Message"] = df["Message"].str.replace('[^\w\s]', '')
    df["Message"] = df["Message"].str.replace(',', '').replace('.', '')
    df['Message'] = df['Message'].str.split()
    x_frame = df.drop("Category", axis=1)
    y_drame = df.drop("Message", axis=1)
    return x_frame, y_drame


if __name__ == '__main__':
    train_X, train_y = process_data("train.csv")
    test_X, test_y = process_data("test.csv")

    classifier = BayesianClassifier()

    user_input = input('test model on database of on a statement? (1 or 2)\n')
    if user_input == '1':
        classifier.fit(train_X, train_y)
        classifier.predict_prob(test_X['Message'][0], test_y['Category'][0])
        print("model score: ", classifier.score(test_X, test_y) * 100, "%")
    if user_input == '2':
        classifier.fit(test_X, test_y)
        classifier.predict_prob(test_X['Message'][0], test_y['Category'][0])
        print('Enter a statement to check:')
        user_message = input()
        message = ' '.join(
            [word for word in user_message.lower().split() if word not in STOP_WORDS])
        message = message.replace(',', '').replace('.', '')
        message_list = message.replace('[^\w\s]', '').split()
        result = classifier.predict(message_list)
        print(result)
