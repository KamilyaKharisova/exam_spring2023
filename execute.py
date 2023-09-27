from datasets.linear_regression_dataset import LinRegDataset
from configs.linear_regression_cfg import cfg
from utils.enums import SetType
from models.linear_regression_model import LinearRegression
import numpy as np

lrdataset = LinRegDataset(cfg)

lrdataset_train = lrdataset(SetType.train)
lrdataset_test = lrdataset(SetType.test)
lrdataset_valid = lrdataset(SetType.valid)
lr = LinearRegression(cfg.base_functions,cfg.lr,cfg.reg_coeff)
lr.train(lrdataset_train['inputs'],lrdataset_train['targets'])
prediction = lr(lrdataset_test['inputs'])
print(np.mean((lrdataset_test['targets'] - prediction)**2))

