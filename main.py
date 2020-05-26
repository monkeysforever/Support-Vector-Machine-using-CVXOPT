from SVM.SVM import SVC
from Utils.Generate_data import generate_dataset
from Utils.Plot_data import plot_dataset

EXAMPLES = 1000
SPLIT_RATIO = 90

X_train, X_test, y_train, y_test = generate_dataset(EXAMPLES,
'blobs', SPLIT_RATIO)

plot_dataset(X_train, y_train)
plot_dataset(X_test, y_test)

y_train = y_train.T
X_train = X_train.T

clf = SVC(C = 0.1, normalize = False)
clf.fit(X_train, y_train)

predictions = clf.predict(X_train)
plot_dataset(X_train.T, predictions.T, clf.w, clf.b)