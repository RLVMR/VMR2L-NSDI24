import os
import json


if __name__ == "__main__":
    num_train = 6000
    num_dev = 200
    num_test = 200

    train_set = set()
    for i in range(num_train):
        with open(f'./data/flex_vm_dataset/M/train/flex_vm_{i}.json', 'r', encoding='utf-8') as f:
            train_set.add(f.read())

    print('train len: ', len(train_set))

    dev_set = set()
    for i in range(num_dev):
        with open(f'./data/flex_vm_dataset/M/dev/flex_vm_{num_train + i}.json', 'r', encoding='utf-8') as f:
            dev_set.add(f.read())

    print('dev len: ', len(dev_set))
    test_set = set()
    for i in range(num_test):
        with open(f'./data/flex_vm_dataset/M/test/flex_vm_{num_train + num_dev + i}.json', 'r', encoding='utf-8') as f:
            test_set.add(f.read())

    print('test len: ', len(test_set))

    print('train + test: ', len(train_set.union(test_set)))
    print('train + dev: ', len(train_set.union(dev_set)))
    print('train + dev + test: ', len(train_set.union(dev_set).union(test_set)))
