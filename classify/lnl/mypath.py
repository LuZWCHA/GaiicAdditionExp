

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'miniimagenet_preset':
            return 'data/mini-imagenet/'
        elif dataset == 'cifar100':
            return 'data/cifar100/'
        elif dataset == 'imagenet32':#The data folder should contain the imagenet32 folder
            return 'data/'
        elif dataset == 'places':
            return 'data/Places/test_256/'
        elif dataset == 'web-bird':
            return 'data/web-bird/'
        elif dataset == 'web-car':
            return 'data/web-car/'
        elif dataset == 'web-aircraft':
            return 'data/web-aircraft/'
        elif dataset == 'custom':
            return 'data/custom/'
        elif dataset == 'ds_custom':
            return "data/train_c2.json", "data/val_c2.json"
        elif dataset == 'ds_custom2':
            return "data/train.json", "data/val.json"
        elif dataset == 'ds_custom3':
            return "data/train_c3.json", "data/val_c3.json"
        else:
            raise NotImplementedError('Dataset {} not available.'.format(dataset))