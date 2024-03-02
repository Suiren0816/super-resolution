#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['./data/dataset',
                                     './data/dataset2',
                                     './data/train'],
                      test_folders=[
                                    './data/testdata',
                                    './data/test'
                                    ],
                      min_size=100,
                      output_folder='./data/')
