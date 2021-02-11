import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculateAggregatedJaccardIndex(groundtruth,prediction):
    C = 0
    U = 0
    Used_Sj = []
    for i in tqdm(np.unique(groundtruth)):
        j=0
        ind_j = 0
        if i>0:
            #for t in np.unique(prediction):
            for t in np.unique((groundtruth==i).astype(np.int) * (prediction).astype(np.int)):
                if ((t>0) and (t not in Used_Sj)):
                    Gi_intersect_Sk = ((groundtruth==i).astype(np.int) * (prediction==t).astype(np.int)).sum()
                    Gi_union_Sk = ((((groundtruth==i).astype(np.int)) + (((prediction==t)).astype(np.int)))>0).astype(np.int).sum()
                    max_k = Gi_intersect_Sk / Gi_union_Sk
                    if max_k > j:
                        j = max_k
                        j_ind = t
                        C = C + abs(Gi_intersect_Sk)
                        U = U + Gi_union_Sk
                        Used_Sj.append(t)
    for i in np.unique(prediction):
        if i > 0:
            if i not in Used_Sj:
                U = U + (prediction==i).astype(np.int).sum()
    #if (U>0):
    #    return C/U
    #else:
    #    return 0
    return C,U

def objectBasedMeasures(groundtruth,prediction):
    FN = FP = TP = US = OS = 0
    FN_MASK = np.zeros((groundtruth.shape[0],groundtruth.shape[1]),dtype=np.int)
    FP_MASK = np.zeros((groundtruth.shape[0],groundtruth.shape[1]),dtype=np.int)
    TP_MASK = np.zeros((groundtruth.shape[0],groundtruth.shape[1]),dtype=np.int)
    US_MASK = np.zeros((groundtruth.shape[0],groundtruth.shape[1]),dtype=np.int)
    OS_MASK = np.zeros((groundtruth.shape[0],groundtruth.shape[1]),dtype=np.int)

    assignment_threshold = 50 # Percent of object to be covered by segmentations to count as detected object
    covered_threshold = 50 # Percent of groundtruth object covers a prediction
    used_prediction_labels = []
    used_gt_labels = []
    for i in tqdm(np.unique(groundtruth)):
        if i>0:
            covered_area = ((groundtruth==i).astype(np.int) * (prediction>0).astype(np.int)).sum()/(groundtruth==i).astype(np.int).sum() * 100
            if covered_area < assignment_threshold:
                FN += 1
                FN_MASK += (groundtruth == i).astype(np.int)
                used_gt_labels.append(i)
            else:
                max_overlap = 0
                max_overlap_pos = 0
                sum_overlap = 0
                OS_MASK_TMP = np.zeros((groundtruth.shape[0], groundtruth.shape[1]), dtype=np.int)
                used_prediction_labels_tmp = []
                for t in np.unique(prediction):
                    if t>0:
                        overlap_gt = ((prediction==t).astype(np.int) * (groundtruth==i).astype(np.int)).sum() / (groundtruth==i).astype(np.int).sum() * 100
                        overlap_prediction = ((prediction==t).astype(np.int) * (groundtruth==i).astype(np.int)).sum() / (prediction==t).astype(np.int).sum() * 100
                        if overlap_gt:
                            used_prediction_labels_tmp.append(t)
                            sum_overlap += 1
                            OS_MASK_TMP += (prediction==t).astype(np.int)
                            if overlap_gt > max_overlap:
                                max_overlap=overlap_gt
                                max_overlap_pos = t
                if sum_overlap > 1:
                    OS += 1
                    OS_MASK += OS_MASK_TMP
                    used_prediction_labels.extend(used_prediction_labels_tmp)
                #else:
                #    TP += 1
                #    TP_MASK += (prediction == max_overlap_pos).astype(np.int)

    for i in tqdm(np.unique(prediction)):
        if i>0:
            covered_area = ((prediction==i).astype(np.int) * (groundtruth>0).astype(np.int)).sum()/(prediction==i).astype(np.int).sum() * 100
            if covered_area < assignment_threshold:
                FP += 1
                FP_MASK += (prediction == i).astype(np.int)
                used_prediction_labels.append(i)
            else:
                max_overlap = 0
                max_overlap_pos = 0
                sum_overlap = 0
                US_MASK_TMP = np.zeros((groundtruth.shape[0], groundtruth.shape[1]), dtype=np.int)
                if (i==25):
                    e=1
                for t in np.unique(groundtruth):
                    if t>0:
                        overlap = ((groundtruth==t).astype(np.int) * (prediction==i).astype(np.int)).sum() / (prediction==i).astype(np.int).sum() * 100
                        if overlap > 0:
                            sum_overlap += 1
                            if overlap > max_overlap:
                                max_overlap=overlap
                                max_overlap_pos = t
                if sum_overlap > 1:
                    US += sum_overlap
                    US_MASK += (prediction==i).astype(np.int)
                    used_prediction_labels.append(i)
    for i in tqdm(np.unique(prediction)):
        if (i>0) and (i not in used_prediction_labels):
            TP = TP + 1
            TP_MASK += (prediction == i).astype(np.int)
    results = dict()
    results["TP"] = TP
    results["FP"] = FP
    results["FN"] = FN
    results["US"] = US
    results["OS"] = OS
    mask = np.zeros((groundtruth.shape[0],groundtruth.shape[1],5))
    mask[:,:,0] = FN_MASK
    mask[:,:,1] = FP_MASK
    mask[:,:,2] = TP_MASK
    mask[:,:,3] = US_MASK
    mask[:,:,4] = OS_MASK
    results["masks"] = mask
    return results

def objectBasedMeasures2(groundtruth,prediction):
    ji_threshold = 0.5
    coveredarea_threshold = 0.5
    tp_mask = np.zeros_like(groundtruth); fn_mask = np.zeros_like(groundtruth); us_mask = np.zeros_like(groundtruth); os_mask = np.zeros_like(groundtruth); fp_mask = np.copy(prediction)
    # Conditions:
    # TP: object must be covered with a prediction with JI from more than ji_threshold; prediction can touch other objects but with maximal object coverage < coveredarea_threshold
    # other predictions can touch the object
    # FN: object is covered by predictions less than covered_area threshold or JI with predictions is less than ji_thresholdbut but predictions that cover the object with more than coveredarea_threshold
    # do not cover another object with > covered_area threshold (otherwise:undersegmentation)
    # FP: prediction covers no object higher than covered_area threshold or has an JI with an object of less than ji_threshold
    # over-segmentation: an object is covered by predictions with > ji_threshold and the predictions does not cover other objects > covered_area threshold

    for i in tqdm(np.unique(groundtruth)):
        if i>0:
            object = groundtruth == i
            # fore each groundtruth object check all predictions overlapping this object
            remaining_objects = object * prediction
            remaining_objects = np.unique(remaining_objects)

            for t in remaining_objects:
                if t>0:
                    ji = jaccardIndex_with_object(object,prediction==t)
                    covered_area = objectarea_covered(object,prediction)
                    # if an object is not covered by at least coveredarea_threshold % than it is not detected --> FN
                    if (covered_area < coveredarea_threshold):
                        fn_mask[np.where(fn_mask == i)] = 0
                        fn_mask += (groundtruth == i) * i
                    else:
                        # if the object is covered for more than coveredarea_threshold % than it might be detected, but must not be
                        # check for prediction if it covers another object with > object_area_threshold
                        objects_coveredbypred = groundtruth * (prediction==t)
                        objects_coveredbypred = np.unique(objects_coveredbypred)
                        count_covered=0
                        index_covered = []
                        pred_involved = []
                        for j in objects_coveredbypred:
                            if j>0:
                                coveredobjectarea = objectarea_covered(groundtruth == j,prediction == t)
                                if coveredobjectarea > coveredarea_threshold:
                                    count_covered += 1
                                    index_covered.append(j)
                                    pred_involved.append(t)
                        if (count_covered == 1) and (ji > ji_threshold):
                            tp_mask += (groundtruth == i) * i
                            fp_mask -= (prediction == pred_involved[0]) * pred_involved[0]
                        elif count_covered > 1: # check for under-segmentation
                            objects_tmp = np.zeros_like(tp_mask)
                            for l in index_covered:
                                objects_tmp += (groundtruth==l)
                            ji_with_objects = jaccardIndex_with_object(objects_tmp>0,prediction == t)
                            if (ji_with_objects>ji_threshold) and (covered_area > coveredarea_threshold):
                                us_mask += (groundtruth == i) * i
                            fn_mask[np.where(fn_mask == i)] = 0
                            fn_mask += (groundtruth == i) * i
                        elif (count_covered == 1) and (ji < ji_threshold):
                            fn_mask[np.where(fn_mask == i)] = 0
                            fn_mask += (groundtruth == i) * i

            # check for over-segmentations
            # only keep predictions in pred_temp that are mainly within the object
            index_covered = []
            for l in remaining_objects:
                coveredpredarea = objectarea_covered(prediction == l, groundtruth == i)
                if coveredpredarea > coveredarea_threshold:
                    index_covered.append(l)
            pred_tmp = np.zeros_like(tp_mask)
            for l in index_covered:
                pred_tmp += (prediction == l)* l
            covered_area = objectarea_covered(object, prediction)
            # check if possibly multiple wrong predictions makeup an object without being an over-segmentation
            if (covered_area>coveredarea_threshold) and (index_covered.__len__() == 0):
                fn_mask[np.where(fn_mask == i)] = 0
                fn_mask += (groundtruth == i) * i

            ji = jaccardIndex_with_area(object, pred_tmp)
            covered_area = objectarea_covered(object, pred_tmp)
            if (ji > ji_threshold) and (remaining_objects.__len__() > 2): # 0 is also always included
                predictions_over_gt = []
                for t in remaining_objects:
                    prediction_over_object_area = objectarea_covered(pred_tmp==t,groundtruth==i)
                    if prediction_over_object_area>coveredarea_threshold:
                        predictions_over_gt.append(t)
                if predictions_over_gt.__len__() > 1:
                    pred_over_gt_mask = np.zeros_like(us_mask)
                    for g in predictions_over_gt:
                        pred_over_gt_mask += (pred_tmp == g)
                    pred_ji = jaccardIndex_with_object(pred_over_gt_mask,groundtruth==i)
                    if pred_ji > ji_threshold:
                        os_mask += (groundtruth == i) * i
                    fn_mask[np.where(fn_mask == i)] = 0
                    fn_mask += (groundtruth == i) * i
            elif (remaining_objects.__len__() > 2) and (covered_area > coveredarea_threshold):
                fn_mask[np.where(fn_mask==i)] = 0
                fn_mask += (groundtruth == i) * i
    # check for FP
    mask = np.zeros((groundtruth.shape[0],groundtruth.shape[1],5))
    mask[:,:,0] = fn_mask
    mask[:,:,1] = fp_mask
    mask[:,:,2] = tp_mask
    mask[:,:,3] = us_mask
    mask[:,:,4] = os_mask
    metrics = getMetrics(mask)
    results = dict()
    results["masks"] = mask
    return results

def getMetrics(mask):
    FN = np.unique(mask[:, :, 0]).__len__() - 1
    FP = np.unique(mask[:, :, 1]).__len__() - 1
    TP = np.unique(mask[:, :, 2]).__len__() - 1
    US = np.unique(mask[:, :, 3]).__len__() - 1
    OS = np.unique(mask[:, :, 4]).__len__() - 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        f1=0
    erg = dict()
    erg["TP"] = TP
    erg["FP"] = FP
    erg["FN"] = FN
    erg["US"] = US
    erg["OS"] = OS
    erg["precision"] = precision
    erg["recall"] = recall
    erg["f1"] = f1
    return erg

def getminObjectSize(mask):
    minsize=100000000
    position=-1
    for i in np.unique(mask):
        if i>0:
            tmp=(mask==i).sum()
            if tmp < minsize:
                minsize=tmp
                position=i
    return minsize-1, position

def jaccardIndex_with_object(object_1,object_2):
    return (object_1 * object_2).sum() / ((object_1 + object_2)>0).sum()

def dicescore_with_object(object_1,object_2):
    return 2*(object_1 * object_2).sum() / (object_1.sum() + object_2.sum())

def jaccardIndex_with_area(object_1,area):
    # Expected: a mask with only an object (object_1)
    #           a labeled mask with multiple objects (area)
    # Returned: the Jaccard Index of the object (object_1) with all touching objects of the labeled mask (area)
    overlapping = object_1 * area;
    area_new = np.zeros_like(area)
    for i in np.unique(overlapping):
        if i>0:
            area_new = area_new + (area==i)
    area_new = area_new > 0
    return (object_1 * area_new).sum() / ((object_1 + area_new)>0).sum()

def objectarea_covered(object,area):
    overlapping = object * area
    area_new = np.zeros_like(area)
    for i in np.unique(overlapping):
        if i>0:
            area_new = area_new + (area==i)
    area_new = area_new > 0
    return (object * area_new).sum() / ((object)>0).sum()

def getMaxJI(object,prediction):
    max_ji = 0
    remaining_objects = object * prediction
    remaining_objects = np.unique(remaining_objects)
    for t in tqdm(remaining_objects):
        if t > 0:
            ji = jaccardIndex_with_object(object,prediction==t)
            if ji > max_ji:
                max_ji = ji
    return ji

def objectBasedMeasures3(groundtruth,prediction):
    ji_threshold = 0.5
    coveredarea_threshold = 0.5
    tp_mask = np.zeros_like(groundtruth); fn_mask = np.zeros_like(groundtruth); us_mask = np.zeros_like(groundtruth); os_mask = np.zeros_like(groundtruth); fp_mask = np.copy(prediction)
    # Conditions:
    # TP: object must be covered with a prediction with JI from more than ji_threshold; prediction can touch other objects but with maximal object coverage < coveredarea_threshold
    # other predictions can touch the object
    # FN: object is covered by predictions less than covered_area threshold or JI with predictions is less than ji_thresholdbut but predictions that cover the object with more than coveredarea_threshold
    # do not cover another object with > covered_area threshold (otherwise:undersegmentation)
    # FP: prediction covers no object higher than covered_area threshold or has an JI with an object of less than ji_threshold
    # over-segmentation: an object is covered by predictions with > ji_threshold and the predictions does not cover other objects > covered_area threshold

    for i in tqdm(np.unique(groundtruth)):
        if i>0:
            object = groundtruth == i
            # if an object is not covered by at least coveredarea_threshold % than it is not detected --> FN
            covered_area = objectarea_covered(object, prediction)
            if (covered_area < coveredarea_threshold):
                fn_mask[np.where(fn_mask == i)] = 0
                fn_mask += (groundtruth == i) * i
            else:
                # if the object is covered for more than coveredarea_threshold % than it might be detected, but must not be
                # check all predictions that touch the object
                remaining_objects = object * prediction
                remaining_objects = np.unique(remaining_objects)

                for t in remaining_objects:
                    if t>0:
                        ji = jaccardIndex_with_object(object,prediction==t)

                        # check for prediction if it covers another object with > object_area_threshold
                        objects_coveredbypred = groundtruth * (prediction==t)
                        objects_coveredbypred = np.unique(objects_coveredbypred)
                        count_covered=0
                        index_covered = []
                        pred_involved = []
                        for j in objects_coveredbypred:
                            if j>0:
                                coveredobjectarea = objectarea_covered(groundtruth == j,prediction == t)
                                if coveredobjectarea > coveredarea_threshold:
                                    count_covered += 1
                                    index_covered.append(j)
                                    pred_involved.append(t)

                        if (count_covered == 1) and (ji > ji_threshold):
                        # if the prediction touches no other object and has a Jaccard index with the object of greater than ji_threshold, than it is a TP
                            tp_mask += (groundtruth == i) * i
                            fp_mask -= (prediction == pred_involved[0]) * pred_involved[0]
                        elif count_covered > 1:
                        # if more objects are touched by the prediction where the area of those objects is covered with more than object_area_threshold, the object it might be part of an under-semgentation
                            objects_tmp = np.zeros_like(tp_mask)
                            for l in index_covered:
                                objects_tmp += (groundtruth==l)
                            ji_with_objects = jaccardIndex_with_object(objects_tmp>0,prediction == t)
                            if (ji_with_objects>ji_threshold) and (covered_area > coveredarea_threshold):
                                us_mask += (groundtruth == i) * i
                            # anyhow, it is a FN
                            fn_mask[np.where(fn_mask == i)] = 0
                            fn_mask += (groundtruth == i) * i
                        elif (count_covered == 1) and (ji < ji_threshold):
                        # if the object is covered by the prediction but the Jaccard index with the prediction is below ji_threshold, that it is a false-negative
                            fn_mask[np.where(fn_mask == i)] = 0
                            fn_mask += (groundtruth == i) * i

                # check for over-segmentations
                # only keep predictions in pred_temp that are mainly within the object
                index_covered = []
                for l in remaining_objects:
                    coveredpredarea = objectarea_covered(prediction == l, groundtruth == i)
                    if coveredpredarea > coveredarea_threshold:
                        index_covered.append(l)
                pred_tmp = np.zeros_like(tp_mask)
                for l in index_covered:
                    pred_tmp += (prediction == l)* l
                covered_area = objectarea_covered(object, prediction)
                # check if possibly multiple wrong predictions makeup an object without being an over-segmentation (predicions are not within the object)
                if (covered_area>coveredarea_threshold) and (index_covered.__len__() == 0):
                    fn_mask[np.where(fn_mask == i)] = 0
                    fn_mask += (groundtruth == i) * i
                else:
                    ji = jaccardIndex_with_area(object, pred_tmp)
                    covered_area = objectarea_covered(object, pred_tmp)
                    if (ji > ji_threshold) and (remaining_objects.__len__() > 2): # 0 is also always included; at least two predictions touch the object
                        predictions_over_gt = []
                        for t in remaining_objects:
                            prediction_over_object_area = objectarea_covered(pred_tmp==t,groundtruth==i)
                            if prediction_over_object_area>coveredarea_threshold:
                                predictions_over_gt.append(t)
                        # at least two predictions touch the object such that each prediction is within the object
                        if predictions_over_gt.__len__() > 1:
                            pred_over_gt_mask = np.zeros_like(us_mask)
                            for g in predictions_over_gt:
                                pred_over_gt_mask += (pred_tmp == g)
                            pred_ji = jaccardIndex_with_object(pred_over_gt_mask,groundtruth==i)
                            if pred_ji > ji_threshold:
                                os_mask += (groundtruth == i) * i
                            fn_mask[np.where(fn_mask == i)] = 0
                            fn_mask += (groundtruth == i) * i
                    elif (remaining_objects.__len__() > 2) and (covered_area > coveredarea_threshold):
                        fn_mask[np.where(fn_mask==i)] = 0
                        fn_mask += (groundtruth == i) * i
    # check for FP
    mask = np.zeros((groundtruth.shape[0],groundtruth.shape[1],5))
    mask[:,:,0] = fn_mask
    mask[:,:,1] = fp_mask
    mask[:,:,2] = tp_mask
    mask[:,:,3] = us_mask
    mask[:,:,4] = os_mask
    metrics = getMetrics(mask)
    results = dict()
    results["masks"] = mask
    return results

def objectBasedMeasures4(groundtruth,prediction,ji_threshold=0.5,coveredarea_threshold=0.5):
    tp_mask = np.zeros_like(groundtruth); fn_mask = np.zeros_like(groundtruth); us_mask = np.zeros_like(groundtruth); os_mask = np.zeros_like(groundtruth); fp_mask = np.copy(prediction)
    object_ji = []
    object_dice = []
    # Conditions:
    # TP: object must be covered with a prediction with JI from more than ji_threshold; prediction can touch other objects but with maximal object coverage < coveredarea_threshold
    # other predictions can touch the object
    # FN: object is covered by predictions less than covered_area threshold or JI with predictions is less than ji_thresholdbut but predictions that cover the object with more than coveredarea_threshold
    # do not cover another object with > covered_area threshold (otherwise:undersegmentation)
    # FP: prediction covers no object higher than covered_area threshold or has an JI with an object of less than ji_threshold
    # over-segmentation: an object is covered by predictions with > ji_threshold and the predictions does not cover other objects > covered_area threshold

    for i in tqdm(np.unique(groundtruth)):
        if i>0:
            object = groundtruth == i
            # if an object is not covered by at least coveredarea_threshold % than it is not detected --> FN
            covered_area = objectarea_covered(object, prediction)
            if ~(covered_area < coveredarea_threshold):
                # if the object is covered for more than coveredarea_threshold % than it might be detected, but must not be
                # check all predictions that touch the object
                remaining_objects = object * prediction
                remaining_objects = np.unique(remaining_objects)

                for t in remaining_objects:
                    if t>0:
                        ji = jaccardIndex_with_object(object,prediction==t)
                        dice = dicescore_with_object(object,prediction==t)

                        # check for prediction if it covers another object with > object_area_threshold
                        objects_coveredbypred = groundtruth * (prediction==t)
                        objects_coveredbypred = np.unique(objects_coveredbypred)
                        count_covered=0
                        index_covered = []
                        pred_involved = []
                        for j in objects_coveredbypred:
                            if j>0:
                                coveredobjectarea = objectarea_covered(groundtruth == j,prediction == t)
                                if coveredobjectarea > coveredarea_threshold:
                                    count_covered += 1
                                    index_covered.append(j)
                                    pred_involved.append(t)

                        if (count_covered == 1) and (ji > ji_threshold):
                        # if the prediction touches no other object and has a Jaccard index with the object of greater than ji_threshold, than it is a TP
                            tp_mask += (groundtruth == i) * i
                            object_ji.append(ji)
                            object_dice.append(dice)
                        # all predictions, that are not true positives, are false positives
                            fp_mask -= (prediction == pred_involved[0]) * pred_involved[0]
                        elif count_covered > 1:
                        # if more objects are touched by the prediction where the area of those objects is covered with more than object_area_threshold, the object it might be part of an under-semgentation
                            objects_tmp = np.zeros_like(tp_mask)
                            for l in index_covered:
                                objects_tmp += (groundtruth==l)
                            ji_with_objects = jaccardIndex_with_object(objects_tmp>0,prediction == t)
                            if (ji_with_objects>ji_threshold) and (covered_area > coveredarea_threshold) and (i in index_covered):
                                us_mask += (groundtruth == i) * i

                # check for over-segmentations
                # only keep predictions in pred_temp that are mainly within the object
                index_covered = []
                for l in remaining_objects:
                    coveredpredarea = objectarea_covered(prediction == l, groundtruth == i)
                    if coveredpredarea > coveredarea_threshold:
                        index_covered.append(l)
                pred_tmp = np.zeros_like(tp_mask)
                for l in index_covered:
                    pred_tmp += ((prediction == l)* l).astype(pred_tmp.dtype)
                covered_area = objectarea_covered(object, prediction)
                # check if possibly multiple wrong predictions makeup an object without being an over-segmentation (predicions are not within the object)
                if ~((covered_area>coveredarea_threshold) and (index_covered.__len__() == 0)):
                    ji = jaccardIndex_with_area(object, pred_tmp)
                    covered_area = objectarea_covered(object, pred_tmp)
                    if (ji > ji_threshold) and (remaining_objects.__len__() > 2): # 0 is also always included; at least two predictions touch the object
                        predictions_over_gt = []
                        for t in remaining_objects:
                            prediction_over_object_area = objectarea_covered(pred_tmp==t,groundtruth==i)
                            if prediction_over_object_area>coveredarea_threshold:
                                predictions_over_gt.append(t)
                        # at least two predictions touch the object such that each prediction is within the object
                        if predictions_over_gt.__len__() > 1:
                            pred_over_gt_mask = np.zeros_like(us_mask)
                            for g in predictions_over_gt:
                                pred_over_gt_mask += (pred_tmp == g)
                            pred_ji = jaccardIndex_with_object(pred_over_gt_mask,groundtruth==i)
                            if pred_ji > ji_threshold:
                                os_mask += (groundtruth == i) * i
    fn_mask = np.copy(groundtruth)
    fn_mask -= tp_mask
    # check for FP
    mask = np.zeros((groundtruth.shape[0],groundtruth.shape[1],5))
    mask[:,:,0] = fn_mask
    mask[:,:,1] = fp_mask
    mask[:,:,2] = tp_mask
    mask[:,:,3] = us_mask
    mask[:,:,4] = os_mask
    metrics = getMetrics(mask)
    results = dict()
    results["masks"] = mask
    results["JI"] = object_ji
    results["DICE"] = object_dice
    return results

def objectBasedMeasuresPerformance(groundtruth,prediction,ji_threshold=0.5,coveredarea_threshold=0.5):
    tp_mask = np.zeros_like(groundtruth); fn_mask = np.zeros_like(groundtruth); us_mask = np.zeros_like(groundtruth); os_mask = np.zeros_like(groundtruth); fp_mask = np.copy(prediction)
    object_ji = []
    object_dice = []
    # Conditions:
    # TP: object must be covered with a prediction with JI from more than ji_threshold; prediction can touch other objects but with maximal object coverage < coveredarea_threshold
    # other predictions can touch the object
    # FN: object is covered by predictions less than covered_area threshold or JI with predictions is less than ji_thresholdbut but predictions that cover the object with more than coveredarea_threshold
    # do not cover another object with > covered_area threshold (otherwise:undersegmentation)
    # FP: prediction covers no object higher than covered_area threshold or has an JI with an object of less than ji_threshold
    # over-segmentation: an object is covered by predictions with > ji_threshold and the predictions does not cover other objects > covered_area threshold

    for i in tqdm(np.unique(groundtruth)):
        if i>0:
            object = groundtruth == i
            # if an object is not covered by at least coveredarea_threshold % than it is not detected --> FN
            covered_area = objectarea_covered(object, prediction)
            if ~(covered_area < coveredarea_threshold):
                # if the object is covered for more than coveredarea_threshold % than it might be detected, but must not be
                # check all predictions that touch the object
                remaining_objects = object * prediction
                remaining_objects = np.unique(remaining_objects)

                for t in remaining_objects:
                    if t>0:
                        ji = jaccardIndex_with_object(object,prediction==t)
                        dice = dicescore_with_object(object,prediction==t)

                        # check for prediction if it covers another object with > object_area_threshold
                        objects_coveredbypred = groundtruth * (prediction==t)
                        objects_coveredbypred = np.unique(objects_coveredbypred)
                        count_covered=0
                        index_covered = []
                        pred_involved = []
                        for j in objects_coveredbypred:
                            if j>0:
                                coveredobjectarea = objectarea_covered(groundtruth == j,prediction == t)
                                if coveredobjectarea > coveredarea_threshold:
                                    count_covered += 1
                                    index_covered.append(j)
                                    pred_involved.append(t)

                        if (count_covered == 1) and (ji > ji_threshold):
                        # if the prediction touches no other object and has a Jaccard index with the object of greater than ji_threshold, than it is a TP
                            tp_mask += (groundtruth == i) * i
                            object_ji.append(ji)
                            object_dice.append(dice)
                        # all predictions, that are not true positives, are false positives
                            fp_mask -= (prediction == pred_involved[0]) * pred_involved[0]
                        elif count_covered > 1:
                        # if more objects are touched by the prediction where the area of those objects is covered with more than object_area_threshold, the object it might be part of an under-semgentation
                            objects_tmp = np.zeros_like(tp_mask)
                            for l in index_covered:
                                objects_tmp += (groundtruth==l)
                            ji_with_objects = jaccardIndex_with_object(objects_tmp>0,prediction == t)
                            if (ji_with_objects>ji_threshold) and (covered_area > coveredarea_threshold) and (i in index_covered):
                                us_mask += (groundtruth == i) * i

                # check for over-segmentations
                # only keep predictions in pred_temp that are mainly within the object
                index_covered = []
                for l in remaining_objects:
                    coveredpredarea = objectarea_covered(prediction == l, groundtruth == i)
                    if coveredpredarea > coveredarea_threshold:
                        index_covered.append(l)
                pred_tmp = np.zeros_like(tp_mask)
                for l in index_covered:
                    pred_tmp += (prediction == l)* l
                covered_area = objectarea_covered(object, prediction)
                # check if possibly multiple wrong predictions makeup an object without being an over-segmentation (predicions are not within the object)
                if ~((covered_area>coveredarea_threshold) and (index_covered.__len__() == 0)):
                    ji = jaccardIndex_with_area(object, pred_tmp)
                    covered_area = objectarea_covered(object, pred_tmp)
                    if (ji > ji_threshold) and (remaining_objects.__len__() > 2): # 0 is also always included; at least two predictions touch the object
                        predictions_over_gt = []
                        for t in remaining_objects:
                            prediction_over_object_area = objectarea_covered(pred_tmp==t,groundtruth==i)
                            if prediction_over_object_area>coveredarea_threshold:
                                predictions_over_gt.append(t)
                        # at least two predictions touch the object such that each prediction is within the object
                        if predictions_over_gt.__len__() > 1:
                            pred_over_gt_mask = np.zeros_like(us_mask)
                            for g in predictions_over_gt:
                                pred_over_gt_mask += (pred_tmp == g)
                            pred_ji = jaccardIndex_with_object(pred_over_gt_mask,groundtruth==i)
                            if pred_ji > ji_threshold:
                                os_mask += (groundtruth == i) * i
    fn_mask = np.copy(groundtruth)
    fn_mask -= tp_mask
    # check for FP
    mask = np.zeros((groundtruth.shape[0],groundtruth.shape[1],5))
    mask[:,:,0] = fn_mask
    mask[:,:,1] = fp_mask
    mask[:,:,2] = tp_mask
    mask[:,:,3] = us_mask
    mask[:,:,4] = os_mask
    metrics = getMetrics(mask)
    results = dict()
    results["masks"] = mask
    results["JI"] = object_ji
    results["DICE"] = object_dice
    return results

def calculateSinglecellDiceJI(groundtruth,prediction):
    object_ji = []
    object_dice = []

    for i in tqdm(np.unique(groundtruth)):
        if i>0:
            object = groundtruth == i
            try:
                remaining_objects = object * prediction
            except:
                e=1
            remaining_objects = np.unique(remaining_objects)

            max_ji= 0
            max_dice= 0
            for t in remaining_objects:
                if t>0:
                    ji = jaccardIndex_with_object(object,prediction==t)
                    dice = dicescore_with_object(object, prediction == t)
                    if ji > max_ji:
                        max_ji = ji
                        max_dice = dice
            object_dice.append(max_dice)
            object_ji.append(max_ji)

    results = dict()
    results["JI"] = object_ji
    results["DICE"] = object_dice
    return results