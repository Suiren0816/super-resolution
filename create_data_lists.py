#!/usr/bin/env python
# mawen mawenmaw-*- encoding: utf-8 -*-


from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['./data/dataset',
                                     './data/dataset2',
                                     './data/dataset3'],
                      test_folders=['./data/test'
                                    ],
                      min_size=100,
                      output_folder='./data/')
