from torchvision import datasets
from MLPclassifier import MLPClassifier

def prepare_data(train: bool = True):
    mnist_train_set = datasets.MNIST(root='./data', train=train, download=True, transform=None)
    print("Downloaded Data")

    x = mnist_train_set.data.numpy()
    x = x.reshape((x.shape[0], -1))
    y = mnist_train_set.targets.numpy()

    return x, y


if __name__ == "__main__":
    x, y = prepare_data()
    layers_dims = [x.shape[1], 20, 30, 20, 10]
    lr = 0.009

    classifier = MLPClassifier(x=x, y=y, learning_rate=lr, layers_dims=layers_dims, batch_size=16)

    print("Start Training")
    classifier.fit()
    print("Finish Training")

    classifier.loss_graph()
    classifier.loss_graph(validation=True)
    classifier.loss_graph(both=True)

    x_test, y_test = prepare_data(train=False)

    test_accuracy = classifier.predict(x=x_test, y=y_test)
    print(f"Test Accuracy: {test_accuracy}")

    train_accuracy = classifier.predict(x=x, y=y)
    print(f"Train Accuracy: {train_accuracy}")
