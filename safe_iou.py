import numpy as np
import argparse
import json
from PIL import Image
from os.path import join


L3_CLASSES = ['road', 'drivable fallback', 'sidewalk', 'non drivable fallback', 'person', 'rider', 'motorcycle', 'bicycle',
                'autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard',
                'traffic sign', 'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky']


class SemanticHierarchy:
    def __init__(self):
        self.d = {}

    def add_class(self, class_name, N1=[], N2=[], N3=[]):
        self.d[class_name] = {'N1': N1, 'N2': N2, 'N3': N3}

    def initialize_hierarchy(self):
        self.add_class('road', N2=['drivable fallback'])
        self.add_class('drivable fallback', N2=['road'])
        self.add_class('sidewalk', N2=['non drivable fallback'])
        self.add_class('non drivable fallback', N2=['sidewalk'])
        self.add_class('person', N2=['rider'])
        self.add_class('rider', N2=['person'])
        self.add_class('motorcycle', N1=['bicycle'], N2=['autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback'])
        self.add_class('bicycle', N1=['motorcycle'], N2=['autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback'])
        self.add_class('autorickshaw', N1=['car'], N2=['bicycle', 'motorcycle', 'truck', 'bus', 'vehicle fallback'])
        self.add_class('car', N1=['autorickshaw'], N2=['bicycle', 'motorcycle', 'truck', 'bus', 'vehicle fallback'])
        self.add_class('truck', N1=['bus', 'vehicle fallback'], N2=['motorcycle', 'bicycle', 'autorickshaw', 'car'])
        self.add_class('bus', N1=['truck', 'vehicle fallback'], N2=['motorcycle', 'bicycle', 'autorickshaw', 'car'])
        self.add_class('vehicle fallback', N1=['truck', 'bus'], N2=['motorcycle', 'bicycle', 'autorickshaw', 'car'])
        self.add_class('curb', N1=['wall'], N2=['fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light'])
        self.add_class('wall', N1=['curb'], N2=['fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light'])
        self.add_class('fence', N1=['guard rail'], N2=['curb', 'wall', 'billboard', 'traffic sign', 'traffic light'])
        self.add_class('guard rail', N1=['fence'], N2=['curb', 'wall', 'billboard', 'traffic sign', 'traffic light'])
        self.add_class('billboard', N1=['traffic sign', 'traffic light'], N2=['curb', 'wall', 'fence', 'guard rail'])
        self.add_class('traffic sign', N1=['billboard', 'traffic light'], N2=['curb', 'wall', 'fence', 'guard rail'])
        self.add_class('traffic light', N1=['billboard', 'traffic sign'], N2=['curb', 'wall', 'fence', 'guard rail'])
        self.add_class('pole', N1=['obs-str-bar-fallback'], N2=['curb', 'wall', 'fence', 'guard rail', 'billboard', 'traffic light', 'traffic sign'])
        self.add_class('obs-str-bar-fallback', N1=['pole'], N2=['curb', 'wall', 'fence', 'guard rail', 'billboard', 'traffic light', 'traffic sign'])
        self.add_class('building', N1=['bridge'], N2=['vegetation'])
        self.add_class('bridge', N1=['building'], N2=['vegetation'])
        self.add_class('vegetation', N2=['bridge', 'building'])
        self.add_class('sky')

    def generate_N3_classes(self, L3_CLASSES):
        for class_name in L3_CLASSES:
            cu_classes = [class_name] + self.d[class_name]['N1'] + self.d[class_name]['N2']
            self.d[class_name]['N3'] = [x for x in L3_CLASSES if x not in cu_classes]

    def get_important_classes(self, IMP_CLASSES, L3_CLASSES):
        ind_of_imp_classes = [i for i, class_name in enumerate(L3_CLASSES) if class_name in IMP_CLASSES]
        return ind_of_imp_classes

    def class_to_id_mapping(self, L3_CLASSES):
        return {class_name: i for i, class_name in enumerate(L3_CLASSES)}

hierarchy = SemanticHierarchy()
hierarchy.initialize_hierarchy()

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist, class_mapping, IMP_CLASSES, ind_of_imp_classes, num_classes=len(L3_CLASSES)):
    ious = []
    safe_ious = []
    den = hist.sum(1) + hist.sum(0) - np.diag(hist)

    for i in range(num_classes):
        current_class = L3_CLASSES[i]
        D = den[i]
        N = hist[i][i]

        T1 = T2 = T3 = penality = 0
        if current_class in IMP_CLASSES:
            for c in [class_mapping[x] for x in hierarchy.d[current_class]['N1']]:
                T1 += hist[i][c]

            for c in [class_mapping[x] for x in hierarchy.d[current_class]['N2']]:
                T2 += hist[i][c]

            for c in [class_mapping[x] for x in hierarchy.d[current_class]['N3']]:
                T3 += hist[i][c]

            W = [1/3, 2/3, 3/3]
            penality = (W[0] * T1) + (W[1] * T2) + (W[2] * T3)
        else:
            for c in [class_mapping[x] for x in hierarchy.d[current_class]['N1']]:
                if c in ind_of_imp_classes:
                    T1 += hist[i][c]

            for c in [class_mapping[x] for x in hierarchy.d[current_class]['N2']]:
                if c in ind_of_imp_classes:
                    T2 += hist[i][c]

            for c in [class_mapping[x] for x in hierarchy.d[current_class]['N3']]:
                if c in ind_of_imp_classes:
                    T3 += hist[i][c]

            W = [1/3, 2/3, 3/3]
            penality = (W[0] * T1) + (W[1] * T2) + (W[2] * T3)

        iou = N / D
        safe_iou = (N - penality) / D

        ious.append(iou)
        safe_ious.append(safe_iou)

    return ious, safe_ious


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, hierarchy, class_mapping, IMP_CLASSES, ind_of_imp_classes, L3_CLASSES, devkit_dir=''):
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = int(info['classes'])
    name_classes = np.array(info['label'], dtype=str)
    mapping = np.array(info['label2train'], dtype=int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val.txt')
    label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = ['_'.join(s[24:].split('/')) for s in pred_imgs]
    pred_imgs = [join(pred_dir, x) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        # print(ind)
        # print(gt_imgs[ind])
        # print(gt_imgs[ind].split('/'))
        if int(gt_imgs[ind].split('/')[7]) in range(0, 206):
            pred = np.array(Image.open(pred_imgs[ind]))
            label = np.array(Image.open(gt_imgs[ind]))
            label = label_mapping(label, mapping)

            if len(label.flatten()) != len(pred.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
                continue

            hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        if ind > 0 and ind % 100 == 0:
            print(ind)

    mIoUs, SmIoUs = per_class_iu(hist, class_mapping, IMP_CLASSES, ind_of_imp_classes, num_classes)
    print(pred_dir)
    pr = [round(x * 100, 0) for x in SmIoUs]
    print(pr)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)) + ':\t' + str(round(SmIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + ':\t' + 'safe miou: ' + str(round(np.nanmean(SmIoUs) * 100, 2)))
    print(f'MIoU of main classes :{str(round(np.nanmean(mIoUs[:len(IMP_CLASSES)]) * 100, 2))} , Safe_MIoU iou of main classes:   {str(round(np.nanmean(SmIoUs[:len(IMP_CLASSES)]) * 100, 2))}')
    print('=========')


def main(args):
    # hierarchy = SemanticHierarchy()
    # hierarchy.initialize_hierarchy()

    # L3_CLASSES = ['road', 'drivable fallback', 'sidewalk', 'non drivable fallback', 'person', 'rider', 'motorcycle', 'bicycle',
    #               'autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard',
    #               'traffic sign', 'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky']

    hierarchy.generate_N3_classes(L3_CLASSES)

    IMP_CLASSES = ['person', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus']

    ind_of_imp_classes = hierarchy.get_important_classes(IMP_CLASSES, L3_CLASSES)

    class_mapping = hierarchy.class_to_id_mapping(L3_CLASSES)

    compute_mIoU(args.gt_dir, args.pred_dir, hierarchy, class_mapping, IMP_CLASSES, ind_of_imp_classes, L3_CLASSES, args.devkit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', default='/ssd_scratch/furqan/Datasets/', type=str, help='directory which stores CityScapes val gt images')
    parser.add_argument('--pred_dir', default='predictions/test', type=str, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='iddaw', help='base directory of cityscapes')
    args = parser.parse_args()
    main(args)
