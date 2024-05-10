def get_train_loader():
    # Loading data
    transform = ToTensor()
    train_set = MNIST(
        root="./../datasets", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)


def get_test_loader():

    test_set = MNIST(
        root="./../datasets", train=False, download=True, transform=transform
    )
