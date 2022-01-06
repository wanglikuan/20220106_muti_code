from FLAlgorithms.trainmodel.MTL_models import MultiLeNetR, MultiLeNetO
import torchvision.models as model_collection
import torch.nn as nn

def get_model(params):
    # data = params['dataset']
    data = params
    if 'mnist' in data:
        model = {}
        model['rep'] = MultiLeNetR()
        # if params['parallel']:
        model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].cuda()
        for i in range(10):
            model[i] = MultiLeNetO()
            model[i].cuda()
        return model

        # if 'L' in params['tasks']:
        #     model['L'] = MultiLeNetO()
        #     if params['parallel']:
        #         model['L'] = nn.DataParallel(model['L'])
        #     model['L'].cuda()
        # if 'R' in params['tasks']:
        #     model['R'] = MultiLeNetO()
        #     if params['parallel']:
        #         model['R'] = nn.DataParallel(model['R'])
        #     model['R'].cuda()
        # return model
