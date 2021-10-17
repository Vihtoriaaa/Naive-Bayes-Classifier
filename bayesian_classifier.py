import pandas as pd


with open('stop_words.txt') as in_file:
    STOP_WORDS = list(in_file.read().split('\n'))


class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.
    """

    def __init__(self):
        self.all_words = set()
        self.data_size = 0

    def fit(self, x, y):
        """
        Fit Naive Bayes parameters according to train data X and y.
        :param X: pd.DataFrame|list - train input/messages
        :param y: pd.DataFrame|list - train output/labels
        :return: None
        """
        df = x.merge(y, left_index=True, right_index=True)
        self.data_size = len(x)
        self.x = x
        self.y = y
        for sms in df['Message']:
            for word in sms:
                self.all_words.add(word)
        word_counter = {unique_word: [
            0] * len(df['Message']) for unique_word in self.all_words}

        for index, sms in enumerate(df['Message']):
            for word in sms:
                word_counter[word][index] += 1
        self.words_df = pd.DataFrame(word_counter)
        self.newdf = pd.concat([df, self.words_df], axis=1)
        self.get_all_probabilities()

    def get_words_from_string(self, input_str):
        message = ' '.join(
            [word for word in input_str.lower().split() if word not in STOP_WORDS])
        message = message.replace(',', '').replace('.', '')
        message_list = message.replace('[^\w\s]', '').split()
        return message_list

    def predict_prob(self, message, label):
        """
        Calculate the probability that a given label can be assigned to a given message.
        :param message: str - input message
        :param label: str - label
        :return: float - probability P(label|message)
        """
        prob = 1

        for word in message:
            try:
                if label == 'spam':
                    prob *= self.parameters_spam[word]
                elif label == 'ham':
                    prob *= self.parameters_ham[word]
            except:
                pass
        return prob

    def get_all_probabilities(self):
        self.spam_mess = self.newdf[self.newdf['Category'] == 'spam']
        self.ham_mess = self.newdf[self.newdf['Category'] == 'ham']
        n_spam = self.spam_mess['Message'].apply(len).sum()
        n_ham = self.ham_mess['Message'].apply(len).sum()
        n_words_vocabulary = len(self.all_words)
        parameters_spam = {
            unique_word: 0 for unique_word in self.all_words}
        parameters_ham = {unique_word: 0 for unique_word in self.all_words}
        for word in self.all_words:
            n_word_given_spam = self.spam_mess[word].sum()
            p_word_given_spam = (n_word_given_spam + 1) / \
                (n_spam + n_words_vocabulary)
            parameters_spam[word] = p_word_given_spam
            n_word_given_ham = self.ham_mess[word].sum()
            p_word_given_ham = (n_word_given_ham + 1) / \
                (n_ham + n_words_vocabulary)
            parameters_ham[word] = p_word_given_ham
        self.parameters_spam = parameters_spam
        self.parameters_ham = parameters_ham

    def predict(self, message):
        """
        Predict label for a given message.
        :param message: str - message
        :return: str - label that is most likely to be truly assigned to a given message
        """
        counter_no_words = 0
        p_ham_given_message, p_spam_given_message = 1, 1
        for word in message:
            if word in self.parameters_spam:
                p_spam_given_message *= self.parameters_spam[word]
            if word not in self.parameters_spam:
                counter_no_words += 1
            if word in self.parameters_ham:
                p_ham_given_message *= self.parameters_ham[word]
            if word not in self.parameters_ham:
                counter_no_words += 1
        if p_spam_given_message > p_ham_given_message:
            return 'spam'
        elif p_spam_given_message < p_ham_given_message:
            return 'ham'
        elif p_spam_given_message == p_ham_given_message:
            if counter_no_words >= len(message)/2:
                return 'needs human classification, probably spam'

    def score(self, X: pd.DataFrame, y: pd. DataFrame) -> float:
        """
        Return the mean accuracy on the given test data
        and labels - the efficiency of a trained model.
        :param X: pd.DataFrame|list - test data - messages
        :param y: pd.DataFrame|list - test labels
        """
        self.total_score = 0

        def test_prediction(row):
            """Tests prediction for the given row."""
            if self.predict(row['Message']) == row['Category']:
                self.total_score += 1
        df = X.merge(y, left_index=True, right_index=True)
        df['predicted'] = df['Message'].apply(
            self.predict)
        df = df.apply(test_prediction, axis=1)
        return self.total_score / len(df)
