import torch

class PaddingCollator :
    def __init__(self, ) :
        pass

    def __call__(self, dataset) :
        diag_tensor = []
        cordi_tensor = []
        rank_tensor = []

        for data in dataset :
            if len(data) == 3 :
                diag, cordi, rank = data
            else :
                diag, cordi = data
                rank = None

            diag_tensor.append(diag)
            cordi_tensor.append(cordi)
            if rank != None :
                rank_tensor.append(rank)

        diag_tensor = torch.tensor(diag_tensor, dtype=torch.float32)
        cordi_tensor = torch.tensor(cordi_tensor, dtype=torch.float32)

        if rank == None :
            batch = {"diag" : diag_tensor, "cordi" : cordi_tensor}
        else :
            rank_tensor = torch.tensor(rank_tensor, dtype=torch.int32)
            batch = {"diag" : diag_tensor, "cordi" : cordi_tensor, "rank" : rank_tensor}
        return batch