
class sharpdataset(object):
    def __init__(self,dataset,datalabel,keylist):
        self.dataset = dataset
        self.labels = datalabel
        self.keys = keylist
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        features = []
        try:
            for i in self.keys:
                features.append(self.dataset[i][idx])
        except KeyError:
            pass
        label = self.labels[idx]
        index = idx
        return features,label,index


