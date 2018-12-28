import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''

        model_dict = {}
        for i in self.labels:
            y_temp = np.zeros((y.shape[0],))
            y_temp[y == i] = 1
            model = svm.LinearSVC(random_state=12345)
            model.fit(X, y_temp)
            model_dict[i] = model

        return model_dict

    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        labels = list(self.labels)
        label_len = len(labels)
        label_pair_list = []
        for i in range(label_len):
            a = labels[i]
            while i + 1 <= label_len - 1:
                label_tup = (a, labels[i + 1])
                label_pair_list.append(label_tup)
                i += 1

        model_dict = {}
        for label_pair in label_pair_list:
            X_temp1 = X[np.where(y == label_pair[0])]
            X_temp2 = X[np.where(y == label_pair[1])]
            X_temp = np.r_[X_temp1, X_temp2]
            y_temp1 = np.ones((X_temp1.shape[0],))
            y_temp2 = np.zeros((X_temp2.shape[0],))
            y_temp = np.r_[y_temp1, y_temp2]
            model = svm.LinearSVC(random_state=12345)
            model.fit(X_temp, y_temp)
            model_dict[label_pair] = model

        return model_dict

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        score_list = []
        for label, model in self.binary_svm.items():
            score = model.decision_function(X)
            score_list.append(score)
        score_array = np.array(score_list).T

        return score_array

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        score_array = np.zeros((X.shape[0], self.labels.shape[0]))
        for label_pair, model in self.binary_svm.items():
            pred = model.predict(X)
            a = np.where(pred == 1)
            b = np.where(pred == 0)
            score_array[a, label_pair[0]] += 1
            score_array[b, label_pair[1]] += 1

        return score_array

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        loss1 = np.sum(np.square(W)) / 2
        sample_size = X.shape[0]
        class_size = W.shape[0]
        all_ones = np.ones((sample_size, class_size))
        eye_array = np.eye(class_size)
        y_one_hot = eye_array[y]
        max_array_temp = (np.dot(W, X.T)).T
        element_array = max_array_temp[np.where(y_one_hot == 1)]
        a = all_ones - y_one_hot + max_array_temp
        max_array = np.max(a, axis=1)
        loss2 = C * np.sum((max_array - element_array))
        loss = loss1 + loss2

        return loss

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        sample_size = X.shape[0]
        class_size = W.shape[0]
        all_ones = np.ones((sample_size, class_size))
        eye_array = np.eye(class_size)
        y_one_hot = eye_array[y]
        max_array_temp = (np.dot(W, X.T)).T
        a = all_ones - y_one_hot + max_array_temp

        max_index = np.argmax(a, axis=1)
        W_grad_list = []
        for i in range(W.shape[0]):
            X_index = np.where(y == np.unique(y)[i])
            X_sample1 = X[X_index]
            Wi_grad1 = np.sum(X_sample1, axis =0)

            X_index2 = np.where(max_index == i)[0]
            if X_index2.shape == (1, 0):
                Wi_grad2 = np.zeros((1, X.shape[1]))
            else:
                X_sample2 = X[X_index2]
                Wi_grad2 = np.sum(X_sample2, axis=0)
            Wi_grad = Wi_grad2 - Wi_grad1
            W_grad_list.append(Wi_grad)
        W_grad_array = np.array(W_grad_list)
        W_grad = W + C * W_grad_array

        return W_grad
