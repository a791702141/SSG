import sys
import numpy as np
sys.path.append('../loader')
from unaligned_data_loader_officehome import UnalignedDataLoaderofficehome
from unaligned_data_loader_office31 import UnalignedDataLoaderoffice31
from officehome import load_data_officehome
from office31 import load_data_office31


officehome = ['art', 'clipart', 'product', 'real_world']
office31 = ['amazon', 'dslr', 'webcam']
def return_dataset_officehome(data, is_target = False, scale=False, usps=False, all_use='no'):
    if data in officehome:
        train_image, train_label, \
        test_image, test_label = load_data_officehome(data, is_target)

    return train_image, train_label, test_image, test_label

def return_dataset_office31(data, is_target = False, scale=False, usps=False, all_use='no'):
    if data in office31:
        train_image, train_label, \
        test_image, test_label = load_data_office31(data)

    return train_image, train_label, test_image, test_label



def dataset_read_officehome(target, batch_size):
    S1 = {}
    S1_test = {}
    S2 = {}
    S2_test = {}
    S3 = {}
    S3_test = {}

    
    S = [S1, S2, S3]
    S_test = [S1_test, S2_test, S3_test]

    T = {}
    T_test = {}
    domain_all = ['art', 'clipart', 'product', 'real_world']
    domain_all.remove(target)
    dataset_size = list()

    target_train, target_train_label , target_test, target_test_label= return_dataset_officehome(target, is_target = True)
    dataset_size.append(target_train.shape[0])
    
    for i in range(len(domain_all)):
        source_train, source_train_label, source_test , source_test_label = return_dataset_officehome(domain_all[i], is_target = False)
        S[i]['imgs'] = source_train
        S[i]['labels'] = source_train_label
        #input target sample when test, source performance is not important
        S_test[i]['imgs'] = source_test
        S_test[i]['labels'] = source_test_label
        dataset_size.append(source_train.shape[0])

    T['imgs'] = target_train
    T['labels'] = target_train_label

    T_test['imgs'] = target_test
    T_test['labels'] = target_test_label

    scale = 224

    train_loader = UnalignedDataLoaderofficehome()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()


    test_loader = UnalignedDataLoaderofficehome()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)
    
    dataset_test = test_loader.load_data()

    return dataset, dataset_test, min(dataset_size)


def dataset_read_office31(target, batch_size):
    S1 = {}
    S1_test = {}
    S2 = {}
    S2_test = {}
    
    S = [S1, S2]
    S_test = [S1_test, S2_test]

    T = {}
    T_test = {}
    domain_all = ['amazon', 'dslr', 'webcam']
    domain_all.remove(target)
    dataset_size = list()

    target_train, target_train_label , target_test, target_test_label= return_dataset_office31(target, is_target = True)
    dataset_size.append(target_train.shape[0])
    
    for i in range(len(domain_all)):
        source_train, source_train_label, source_test , source_test_label = return_dataset_office31(domain_all[i], is_target = False)
        S[i]['imgs'] = source_train
        S[i]['labels'] = source_train_label
        #input target sample when test, source performance is not important
        S_test[i]['imgs'] = source_test
        S_test[i]['labels'] = source_test_label
        dataset_size.append(source_train.shape[0])


    T['imgs'] = target_train
    T['labels'] = target_train_label

    T_test['imgs'] = target_test
    T_test['labels'] = target_test_label

    scale = 224

    train_loader = UnalignedDataLoaderoffice31()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()


    test_loader = UnalignedDataLoaderoffice31()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)
    
    dataset_test = test_loader.load_data()

    return dataset, dataset_test, min(dataset_size)
