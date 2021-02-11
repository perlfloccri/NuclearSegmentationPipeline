function [fscore,AJI] = score(mask,groundtruth)
    % AJI value initialization
    C = 0;
    U = 0;
    Used_Sj = [];
    AJI = 0;
    % F1 value initialization
    FN = 0;
    FP = 0;
    TP = 0;
    REC=0;
    PREC=0;
    fscore=0;
    sav_mask_indizes = [];
    unique_values = unique(groundtruth(:));

    for i = 2:size(unique_values)
        AJI_j=0;
        AJI_ind_j = 0;
        actual_val = unique_values(i);
        gt_tmp = (groundtruth == actual_val);
        tmp = uint8(mask) .* uint8(gt_tmp);
        remaining = unique(tmp(:));
        if length(remaining) == 1
            FN = FN + 1;
        end
        rem = 0;
        rem_label = -1;
        for x = 2:size(remaining)
            actual_mask = remaining(x);
            IoUnenner = sum(sum((uint8(mask == actual_mask) .* uint8(gt_tmp))>0));
            IoUzaehler = sum(sum(((uint8(mask == actual_mask) + uint8(gt_tmp))>0)));
            IoU = IoUnenner / IoUzaehler;
            if IoU > 0.5
                rem = 1;
                rem_label = actual_mask;
            end
            % AJI
            if sum(ismember(Used_Sj,actual_mask)) == 0
                max_k = IoU;
                if max_k > AJI_j
                    AJI_j = max_k;
                    AJI_ind_j = actual_mask;
                    C = C + abs(IoUnenner);
                    U = U + IoUzaehler;
                    Used_Sj = [Used_Sj;actual_mask];
                end
            end
            
        end
        if rem == 1
            TP = TP + 1;
            sav_mask_indizes = [sav_mask_indizes;rem_label];
        end
        
    end
    unique_predictions = unique(mask);
    for i =2:size(unique_predictions,1)
        if sum(ismember(Used_Sj,i)) == 0
                U = U + sum(sum(mask==i));
        end
    end
    if U>0
        AJI = C / U;
    end
    FP = length(unique(mask(:)))-1 - length(sav_mask_indizes);
    if (TP+FP) > 0
        PREC = TP / (TP + FP);
    end
    if (TP + FN) > 0
        REC = TP / (TP + FN);
    end
    if (PREC + REC) > 0 
        fscore = 2 * (PREC * REC) / (PREC + REC);
    end
end