import unittest
import numpy as np
from M1.AdaPoinTr.src.entrenamiento import cloud_metrics

class PilotMetricsTest(unittest.TestCase):
    def test_identical_cloud_is_exact(self):
        p=np.array([[0.,0,0],[2.,0,0]])
        result=cloud_metrics(p,p)
        self.assertEqual(result['cd2'],0.)
        self.assertEqual(result['fscore'],1.)

    def test_partial_cloud_has_perfect_precision_but_incomplete_recall(self):
        reference=np.array([[0.,0,0],[2.,0,0]])
        result=cloud_metrics(reference[:1],reference)
        self.assertEqual(result['precision'],1.)
        self.assertEqual(result['recall'],.5)
        self.assertAlmostEqual(result['fscore'],2/3)
        self.assertEqual(result['cd_l1'],.5)
        self.assertEqual(result['cd2'],2.)

    def test_far_prediction_has_no_matches(self):
        result=cloud_metrics(np.array([[10.,0,0]]),np.array([[0.,0,0]]))
        self.assertEqual(result['fscore'],0.)
        self.assertEqual(result['cd_l1'],10.)
        self.assertEqual(result['cd2'],200.)

if __name__=='__main__':unittest.main()
