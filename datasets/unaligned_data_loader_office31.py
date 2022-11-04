import torch.utils.data
# import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms
from datasets_office31 import Dataset


class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, data_loader_t, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.data_loader_t = data_loader_t
     
        self.stop_A = False
        self.stop_B = False
        self.stop_t = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.stop_t = False

        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.data_loader_t_iter = iter(self.data_loader_t)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        B, B_paths = None, None
        t, t_paths = None, None
        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)

        try:
            B, B_paths = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        try:
            t, t_paths = next(self.data_loader_t_iter)
        except StopIteration:
            if t is None or t_paths is None:
                self.stop_t = True
                self.data_loader_t_iter = iter(self.data_loader_t)
                t, t_paths = next(self.data_loader_t_iter)

        if (self.stop_A and self.stop_B and self.stop_t) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            self.stop_t = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S1': A, 'S1_label': A_paths,
                    'S2': B, 'S2_label': B_paths,
                    'T': t, 'T_label': t_paths}


class UnalignedDataLoaderoffice31():
    def initialize(self, source, target, batch_size1, batch_size2, scale=256):
        transform = transforms.Compose([
            transforms.Resize([scale, scale]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        #dataset_source1 = Dataset(source[1]['imgs'], source['labels'], transform=transform)
        dataset_source1 = Dataset(source[0]['imgs'], source[0]['labels'], transform=transform)
        data_loader_s1 = torch.utils.data.DataLoader(dataset_source1, batch_size=batch_size1, shuffle=True, num_workers=1)
        self.dataset_s1 = dataset_source1

        dataset_source2 = Dataset(source[1]['imgs'], source[1]['labels'], transform=transform)
        data_loader_s2 = torch.utils.data.DataLoader(dataset_source2, batch_size=batch_size1, shuffle=True, num_workers=1)
        self.dataset_s2 = dataset_source2     

        dataset_target = Dataset(target['imgs'], target['labels'], transform=transform)
        data_loader_t = torch.utils.data.DataLoader(dataset_target, batch_size=batch_size2, shuffle=True, num_workers=1)
        

        self.dataset_t = dataset_target
        self.paired_data = PairedData(data_loader_s1, data_loader_s2, data_loader_t,
                                      float("inf"))
       

    def name(self):
        return 'UnalignedDataLoaderoffice31'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_s1),len(self.dataset_s2), len(self.dataset_t)), float("inf"))


