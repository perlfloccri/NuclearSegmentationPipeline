%* ___inputName___	:	The name of the image file. It can be an RGB or a grayscale image.
%* ___tArea___	:	The area threshold.
%* ___dSize___ :	The size of the disk structuring elements and the average filter.
%* ___gAngle___ :	The	angle to define the start and end points of an arc, whose pixels are used to define the stopping condition of the flooding process.
%* ___offset___ :	The offset is the maximum number of pixels that a marker grows at the end without considering the stopping condition.
%* ___segmRes___ :	The output segmentation map. Pixels of the same cell nucleus are labeled with the same integer, where labels are from 1 to N. Background pixels are labeled with 0.
addpath('C:\Users\flo\Desktop\IterativeHMinima-master');
plot_results = 1;
name = "tiles_scaled_silver";
% Non-scaled samples
% 20x :
%results = ["tarea" "dSize" "gAngle" "offset" "F1 score" "AJI"];
results = [];
tarea = [1,2,3];
dSize = [11,12,13];
gAngle = [14,15,16];
offset = [1,2,3];
total_combinations = length(tarea) * length(dSize) * length(gAngle) * length(offset);
imgpath = ['C:\dataset\data\' + name + '\train\' 'C:\dataset\data\' + name + '\val\'];
imagefiles = [];
for i=1:length(imgpath)
   diagnosis = dir(imgpath(i)); 
   for x=3:length(diagnosis)
       curr_files = dir(strcat(imgpath(i),diagnosis(x).name + "\images\", '*.jpg'));
       for t=1:length(curr_files)
            imagefiles = [imagefiles; imgpath(i) + diagnosis(x).name + "\images\" + curr_files(t).name]; 
       end
   end
end
%results = iterativeHmin('gnb.png', 20, 10, 5, 2);

indizes = 1:length(imagefiles);
totalcount = 0;
for a = tarea
    for b = dSize
        for c=gAngle
            for d=offset
                total_f1_score = [];
                total_AJI = [];
                cnt = 0;
                iwant = indizes(randperm(length(indizes),15));
                for i = iwant
                    img = imagefiles(i);
                    groundtruth = imread(strrep(strrep(strrep(imagefiles(i),'.jpg','.tif'),'Img_','Mask_'),'images','masks'));
                    if length(unique(groundtruth(:)))>0
                        prediction = iterativeHmin(img, a, b, c, d);
                        [f1_score,AJI] = score(prediction,groundtruth);
                        total_f1_score = [total_f1_score;f1_score];
                        total_AJI = [total_AJI;AJI];
                        cnt = cnt + 1;
                        %figure(1),imshow(groundtruth>0,[]);figure(2),imshow(prediction,[]);
                        %disp("Parameters : " + string(a) + " " + string(b) + " " + string(c) + " " + string(d))
                        %disp ("Image " + imagefiles(i).name + ", F1-Score: " + string(f1_score) + ", AJI-Score: " + string(AJI))
                    end
                end
                if cnt > 0
                    disp("Mean score over " + string(cnt) + " samples: F1-score=" + string(mean(total_f1_score)) + " +/- " + string(std(total_f1_score)) + ", AJI=" + string(mean(total_AJI)) + " +/- " + string(std(total_AJI)))
                    totalcount = totalcount + 1;
                    disp("Iteration " + string(totalcount) + " from " + string(total_combinations))
                    results = [results; a b c d mean(total_f1_score) std(total_f1_score) mean(total_AJI) std(total_AJI)];
                end
            end
        end
    end
end
foldername = "iterativehmin\results_" + name;
mkdir (foldername);
if (plot_results)
    max_x = 400;
    FigH = figure('Position', get(0, 'Screensize'));plot(results(:,1),'g-','LineWidth',2);title('tArea');%xlim([0 max_x]);
    F = getframe(FigH);imwrite(F.cdata, foldername + '\tArea_fine.png', 'png');
    close(FigH)
    FigH = figure('Position', get(0, 'Screensize')); plot(results(:,2),'m-','LineWidth',2);title('dSize');%xlim([0 max_x]);
    F = getframe(FigH);imwrite(F.cdata, foldername + '\dSize_fine.png', 'png');
    close(FigH)
    FigH = figure('Position', get(0, 'Screensize'));plot(results(:,3),'r-','LineWidth',2);title('gAngle');%xlim([0 max_x]);
    F = getframe(FigH);imwrite(F.cdata, foldername + '\gAngle_fine.png', 'png');
    close(FigH)
    FigH = figure('Position', get(0, 'Screensize'));plot(results(:,4),'b-','LineWidth',2);title('offset');%xlim([0 max_x]);
    F = getframe(FigH);imwrite(F.cdata, foldername + '\offset_fine.png', 'png');
    close(FigH)
    FigH = figure('Position', get(0, 'Screensize')),jbfill(1:length(results(:,5)),(results(:,5) - results(:,6))',(results(:,5) + results(:,6))',[0.9 0.4 0.4],[0.9 0.5 0.5],0,1);%xlim([0 max_x]);
    hold on; plot(1:length(results(:,5)), results(:,5),'k-','LineWidth',2);title("F1 score");
    F = getframe(FigH);imwrite(F.cdata, foldername + '\F1_fine.png', 'png');
    close(FigH)
    FigH = figure('Position', get(0, 'Screensize')),jbfill(1:length(results(:,7)),(results(:,7) - results(:,8))',(results(:,7) + results(:,8))',[0.4 0.9 0.4],[0.5 0.9 0.5],0,1);%xlim([0 max_x]);
    hold on; plot(1:length(results(:,7)), results(:,7),'k-','LineWidth',2);title("AJI");
    F = getframe(FigH);imwrite(F.cdata, foldername + '\AJI_fine.png', 'png');
    close(FigH)
end

% Save results
combined_results = (results(:,5) - results(:,6)) + (results(:,7) - results(:,8)) - abs(results(:,5)-results(:,6) - results(:,7) - results(:,8));
max_pos = find(combined_results==max(combined_results));
erg = results(max_pos,:); % Best parameter combination
save(foldername + "\results_fine.mat", "results");
save(foldername + "\combined_results_fine.mat", "combined_results");
save(foldername + "\results_chosen_fine.mat", "erg");
save(foldername + "\position_max_fine.mat", "max_pos");

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