'''
    Evaluators for semantic segmentation, working in-mem from instances.

    2021 Benjamin Kellenberger
'''

from typing import OrderedDict
import torch
from detectron2.evaluation import DatasetEvaluator
from detectron2.data import MetadataCatalog

from engine import util


class SemSegEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, output_dir=None, class_offset=1, recall_minNumPixels=(1, 50, 100, 500, 1000)):
        super(SemSegEvaluator, self).__init__()
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.class_offset = class_offset
        self.recall_minNumPixels = recall_minNumPixels
        self._cpu_device = torch.device('cpu')
        self.num_classes = len(MetadataCatalog.get(self.dataset_name).thing_classes) + self.class_offset
        self.classNames = list(['background_'+str(o) for o in range(self.class_offset)]) + MetadataCatalog.get(self.dataset_name).thing_classes
        self.reset()


    def reset(self):
        # instance-specific stats
        self._recalls_numPixels = {}
        self._numSamples_numPixels = {}
        for value in self.recall_minNumPixels:
            self._recalls_numPixels[value] = dict.fromkeys(list(range(self.num_classes)), 0.0)
            self._numSamples_numPixels[value] = dict.fromkeys(list(range(self.num_classes)), 0.0)

        # general stats
        self._oa = 0.0
        self._precisions = dict.fromkeys(list(range(self.num_classes)), 0.0)
        self._recalls = dict.fromkeys(list(range(self.num_classes)), 0.0)
        self._num_samples = 0

    
    def evaluate(self):
        #TODO: logger; saving
        results = OrderedDict()

        # general stats for all instances
        resultStr = 'General stats\n'
        oa = self._oa / self._num_samples
        results['OA'] = {'OA': oa}
        resultStr += 'OA:   {:.4f}\n'.format(oa)
        resultStr += '| Class | Precision | Recall |\n| ----- | --------- | ------ |\n'
        results['prec'] = {}
        results['rec'] = {}
        for cl in range(self.num_classes):
            prec = self._precisions[cl] / self._num_samples
            rec = self._recalls[cl] / self._num_samples
            resultStr += '| {} | {:.4f} | {:.4f} |\n'.format(
                self.classNames[cl], prec, rec
            )
            results['prec'][self.classNames[cl]] = prec
            results['rec'][self.classNames[cl]] = rec
        
        # instance-specific stats
        resultStr += '\n\nObject recall at different levels of intersection (no. pixels)\n'
        resultStr += '| Class | Min Area | No. targets | Recall |\n| ----- | -------- | ----------- | ------ |\n'
        for value in self.recall_minNumPixels:
            for cl in range(self.num_classes):
                rec = 100 * self._recalls_numPixels[value][cl] / self._numSamples_numPixels[value][cl]
                resultStr += '| {} | {} | {} | {:.4f} |\n'.format(
                    self.classNames[cl], value, self._numSamples_numPixels[value][cl], rec
                )

        print(resultStr)
        return resultStr


    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            sz = torch.tensor(input['image'].size())[-2:]
            yhat = torch.argmax(output.squeeze(), 0).to(self._cpu_device)

            # instance-specific stats
            for i in range(len(input['instances'])):
                target = util.instances_to_segmask(input['instances'][i], (sz[0],sz[1]), self.class_offset).to(self._cpu_device)
                for value in self._recalls_numPixels:
                    for cl in range(self.num_classes):
                        if sum(sum(target==cl)).item() >= value:
                            # only evaluate at given no. pixels level if target object is large enough
                            self._numSamples_numPixels[value][cl] += 1
                            self._recalls_numPixels[value][cl] += (sum(sum((target == cl) * (yhat == cl))).item() >= value)

            # general stats for all instances
            target = util.instances_to_segmask(input['instances'], (sz[0],sz[1]), self.class_offset).to(self._cpu_device)
            yhat = torch.argmax(output.squeeze(), 0).to(self._cpu_device)
            self._oa += torch.mean((target == yhat).float()).item()
            for cl in range(self.num_classes):
                tp = sum(sum((target == cl) * (yhat == cl))).item()
                fp = sum(sum((target != cl) * (yhat == cl))).item()
                fn = sum(sum((target == cl) * (yhat != cl))).item()
                try:
                    prec = tp / (tp + fp)
                except:
                    prec = 0.0
                self._precisions[cl] += prec
                
                try:
                    rec = tp / (tp + fn)
                except:
                    rec = 0.0
                self._recalls[cl] += rec
            self._num_samples += 1