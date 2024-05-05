from monai.transforms.transform import MapTransform
import torch 

class ProgressiveSoftEncode(MapTransform):

    def __call__(self, data):
        for key in self.keys:
            label = data[key]
            if int(label) == 0:
                data[key]= torch.FloatTensor([0.95, 0.05, 0])
            elif int(label) == 1:
                data[key]= torch.FloatTensor([0.05, 0.9, 0.05])
            elif int(label) == 2:
                data[key]= torch.FloatTensor([0, 0.05, 0.95])
        return data