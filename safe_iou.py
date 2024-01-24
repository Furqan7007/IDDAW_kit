import numpy as np
import argparse
import json
from PIL import Image
from os.path import join


d = {}

d['road'] = {'N1':[], 'N2':['drivable fallback'], 'N3':[]}

d['drivable fallback'] = {'N1':[], 'N2':['road'], 'N3':[]}

d['sidewalk'] = {'N1':[], 'N2':['non drivable fallback'], 'N3':[]}

d['non drivable fallback'] = {'N1':[], 'N2':['sidewalk'], 'N3':[]}

d['person'] = {'N1':[], 'N2':['rider'], 'N3':[]}

d['rider'] = {'N1':[], 'N2':['person'], 'N3':[]}

d['motorcycle'] = {'N1':['bicycle'], 'N2':['autorickshaw', 'car', 'truck',

                            'bus', 'vehicle fallback'], 'N3':[]}

d['bicycle'] = {'N1':['motorcycle'], 'N2':['autorickshaw', 'car', 'truck',

                            'bus', 'vehicle fallback'], 'N3':[]}

d['autorickshaw'] = {'N1':['car'], 'N2':['bicycle', 'motorcycle', 'truck',

                            'bus', 'vehicle fallback'], 'N3':[]}

d['car'] = {'N1':['autorickshaw'], 'N2':['bicycle', 'motorcycle', 'truck',

                            'bus', 'vehicle fallback'], 'N3':[]}

d['truck'] = {'N1':['bus', 'vehicle fallback'], 'N2':['motorcycle',

                            'bicycle', 'autorickshaw', 'car'], 'N3':[]}

d['bus'] = {'N1':['truck', 'vehicle fallback'], 'N2':['motorcycle',

                            'bicycle', 'autorickshaw', 'car'], 'N3':[]}

d['vehicle fallback'] = {'N1':['truck', 'bus'], 'N2':['motorcycle',

                            'bicycle', 'autorickshaw', 'car'], 'N3':[]}

d['curb'] = {'N1':['wall'], 'N2':['fence', 'guard rail', 'billboard',

                            'traffic sign', 'traffic light'], 'N3':[]}

d['wall'] = {'N1':['curb'], 'N2':['fence', 'guard rail', 'billboard',

                            'traffic sign', 'traffic light'], 'N3':[]}

d['fence'] = {'N1':['guard rail'], 'N2':['curb', 'wall', 'billboard',

                            'traffic sign', 'traffic light'], 'N3':[]}

d['guard rail'] = {'N1':['fence'], 'N2':['curb', 'wall', 'billboard',

                            'traffic sign', 'traffic light'], 'N3':[]}

d['billboard'] = {'N1':['traffic sign', 'traffic light'], 'N2':['curb',

                            'wall', 'fence', 'guard rail'], 'N3':[]}

d['traffic sign'] = {'N1':['billboard', 'traffic light'], 'N2':['curb',

                            'wall', 'fence', 'guard rail'], 'N3':[]}

d['traffic light'] = {'N1':['billboard', 'traffic sign'], 'N2':['curb',

                            'wall', 'fence', 'guard rail'], 'N3':[]}

d['pole'] = {'N1':['obs-str-bar-fallback'], 'N2':['curb', 'wall', 'fence', 
                                                  'guard rail', 'billboard', 'traffic light', 'traffic sign'], 'N3':[]}

d['obs-str-bar-fallback'] = {'N1':['pole'], 'N2':['curb', 'wall', 'fence',

                             'guard rail', 'billboard', 'traffic light', 'traffic sign'], 'N3':[]}

d['building'] = {'N1':['bridge'], 'N2':['vegetation'], 'N3':[]}

d['bridge'] = {'N1':['building'], 'N2':['vegetation'], 'N3':[]}

d['vegetation'] = {'N1':[], 'N2':['bridge', 'building'], 'N3':[]}

d['sky'] = {'N1':[], 'N2':[], 'N3':[]}

# only classes in IMP_CLASSES are penalised. modify list for experimentation

IMP_CLASSES = ['person', 'rider', 'motorcycle', 'bicycle', 'autorickshaw',

           'car', 'truck', 'bus' ]  # traffic participants

# IMP_CLASSES = ['road', 'drivable fallback', 'sidewalk', 'non drivable fallback',

#            'person', 'rider', 'motorcycle', 'bicycle', 'autorickshaw',

#            'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall',

#            'fence', 'guard rail', 'billboard', 'traffic sign',

#            'traffic light']


#L3_CLASSES are level 3 classes in label hierearchy 
L3_CLASSES = ['road', 'drivable fallback', 'sidewalk', 'non drivable fallback',

           'person', 'rider', 'motorcycle', 'bicycle', 'autorickshaw',

           'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall',

           'fence', 'guard rail', 'billboard', 'traffic sign',

           'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky']

print(f'number of important classes : {len(IMP_CLASSES)}/{len(L3_CLASSES)}')

ind_of_imp_classes=[]
for i in range(len(L3_CLASSES)):
    if  L3_CLASSES[i] in IMP_CLASSES:
        ind_of_imp_classes.append(i)
print(f"Index of important classes : {ind_of_imp_classes}")
#To convert class name to class id
class2id = {}
for i in range(len(L3_CLASSES)):
    class2id[L3_CLASSES[i]] = i

# add classes that belongs to N3(td=3) to the dictonary
for i in range(len(L3_CLASSES)):
    cu_classes = [L3_CLASSES[i]] +  d[L3_CLASSES[i]]['N1']+d[L3_CLASSES[i]]['N2']  # self class, N1&N2 Classes
    d[L3_CLASSES[i]]['N3'] = [x for x in L3_CLASSES if x not in cu_classes]  # assigning N3 classes L3_CLASSES - [CURRENT CLASS , N1 & N2 classes ]


def fast_hist(a, b, n):
    '''
    input : flattened label and flattened pred 
    output : confusion matrix for that image; size : nXn 
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist,num_classes=len(L3_CLASSES)):
    '''
    input : nXn array, confusion matrix
    output : return iou list and safe iou list (lsit contain iou for each class)
    safe miou =miou, if class is not in IMP classes
    safe miou =miou - penality , if class is in IMP classes
    
    '''
    ious =[]
    safe_ious =[]
    den = hist.sum(1) + hist.sum(0) - np.diag(hist) #denominators (unious)

    for i in range(num_classes):    
        current_class = L3_CLASSES[i]
        D=den[i]
        N=hist[i][i]

        T1=0
        T2=0
        T3=0
        penality = 0
        if current_class in IMP_CLASSES:  
        
            for c in [class2id[x] for x in d[current_class]['N1']]: # index of class with td=1 with current class
                T1+=hist[i][c]

            for c in [class2id[x] for x in d[current_class]['N2']]: # index of class with td=1 with current class
                T2+=hist[i][c]

            for c in [class2id[x] for x in d[current_class]['N3']]: # index of class with td=3 with current class
                T3+=hist[i][c]
        
            W = [1/3 , 2/3 , 3/3] # td = [1, 2, 3] #treedistance/2 ; weight is td/L where L is number of levels in hierarchy

            penality = (W[0]*T1) + (W[1]*T2) + (W[2]*T3)

        else:
            
            for c in [class2id[x] for x in d[current_class]['N1']]: # index of class with td=1 with current class
                if c in ind_of_imp_classes:
                    T1+=hist[i][c]

            for c in [class2id[x] for x in d[current_class]['N2']]: # index of class with td=1 with current class
                if c in ind_of_imp_classes:
                    T2+=hist[i][c]

            for c in [class2id[x] for x in d[current_class]['N3']]: # index of class with td=3 with current class
                if c in ind_of_imp_classes:
                    T3+=hist[i][c]
        
            W = [1/3 , 2/3 , 3/3] # td = [1, 2, 3] #treedistance/2 ; weight is td/L where L is number of levels in hierarchy

            penality = (W[0]*T1) + (W[1]*T2) + (W[2]*T3)

        # if i>=18:
        #     print(f'{i} : {hist[i]} ')
        
        iou = N/D # intersection over uniou for current class
        safe_iou = (N - penality)/D

        ious.append(iou)
        safe_ious.append(safe_iou)
    return ious, safe_ious


def label_mapping(input, mapping):
    "if mapping is different use label_mapping to map labels"
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the ground truth directory and pred directory 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = int(info['classes'])
    #print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=str)
    #print(name_classes)
    mapping = np.array(info['label2train'], dtype=int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val.txt')
    label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()

    pred_imgs = ['_'.join(s[24:].split('/')) for s in pred_imgs]

    pred_imgs = [pred_dir+'/'+ x for x in pred_imgs]
    for ind in range(len(gt_imgs)):
        if int(gt_imgs[ind].split('/')[6]) in range(0,206): #[39,55,163,197]:#range(178,205): # 46,106,178 to get class wise by giving folder sequence in range 
            pred=np.array(Image.open(pred_imgs[ind]))
            label=np.array(Image.open(gt_imgs[ind]))
            #pred = np.array(pr.resize((1920,1080)) ) 
            #label = np.array(la.resize((1920,1080)))
            label = label_mapping(label, mapping)
            if len(label.flatten()) != len(pred.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
                continue
            #histpm = fast_hist(label.flatten(), pred.flatten(), num_classes)
            hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        if ind > 0 and ind % 100 == 0:
            print(ind) #print(' {:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist,num_classes)[0])))


    mIoUs, SmIoUs = per_class_iu(hist,num_classes) 
    print(pred_dir)
    pr = [round(x * 100, 0) for x in SmIoUs]
    print(pr)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)) + ':\t' + str(round(SmIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + ':\t' + 'safe miou: ' + str(round(np.nanmean(SmIoUs) * 100, 2)))
    print(f'MIoU of main classes :{str(round(np.nanmean(mIoUs[:len(IMP_CLASSES)]) * 100, 2))} , Safe_MIoU iou of main classes:   {str(round(np.nanmean(SmIoUs[:len(IMP_CLASSES)]) * 100, 2))}')
    print('=========')
    return  mIoUs, SmIoUs 


def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)

# pths = ['predictions_heir/predictions_internimage_b_heirloss_240k' ,
#         'predictions_heir/predictions_internimage_s_iddawpretrain_heirloss_272k' ,
#         'predictions_heir/predictions_internimage_s_iddpretrain_heirloss_224k',
#         'predictions_old/predictions_internimage_s_old',
#         'predictions_old/predictions_internimage_b_old']

pths = ['predictions/predictions_train_b_iou_loss']
for p in pths :
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--gt_dir', default='/scratch/nikhil/',type=str, help='directory which stores CityScapes val gt images')
        parser.add_argument('--pred_dir', default=p, type=str, help='directory which stores CityScapes val pred images')
        parser.add_argument('--devkit_dir', default='iddaw', help='base directory of cityscapes')
        args = parser.parse_args()
        main(args)