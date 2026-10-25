import unittest
import numpy as np
import torch
from M1.AdaPoinTr.src.entrenamiento import calibrate_bn


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn=torch.nn.BatchNorm1d(3,momentum=.9)
        self.root_modes=[]

    def forward(self,x):
        self.root_modes.append(self.training)
        if self.training:
            raise AssertionError('La calibracion no debe activar las consultas de denoising')
        return self.bn(x.transpose(1,2))


class CalibrationTest(unittest.TestCase):
    def test_updates_statistics_without_weights_or_training_branch(self):
        model=Probe();weights={k:v.clone() for k,v in model.named_parameters()}
        points=np.arange(24,dtype=np.float32).reshape(8,3)
        calibrate_bn(model,[{'partial':points}],batch_size=1)
        self.assertEqual(model.root_modes,[False])
        self.assertFalse(model.training);self.assertFalse(model.bn.training)
        self.assertEqual(model.bn.momentum,.9)
        torch.testing.assert_close(model.bn.running_mean,torch.from_numpy(points.mean(axis=0)))
        for k,v in model.named_parameters():torch.testing.assert_close(v,weights[k])

    def test_reference_not_required(self):
        model=Probe()
        calibrate_bn(model,[{'partial':np.ones((8,3),dtype=np.float32)}])
        self.assertEqual(model.bn.num_batches_tracked.item(),1)


if __name__=='__main__':unittest.main()
