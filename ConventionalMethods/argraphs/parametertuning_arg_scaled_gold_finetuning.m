addpath('z:\ARGraphs\x64\Release');
addpath('z:\ARGraphs');
plot_results = 1;
name = 'tiles_scaled_gold';
% Non-scaled samples
% 20x :
%results = ['tarea' 'dSize' 'gAngle' 'offset' 'F1 score' 'AJI'];
results = [];
Tsize = [15,16,17];
Tperc = [0.3,0.35 0.4];
Tstd = [6,7,8];
fW = [7,8,9];
total_combinations = length(Tsize) * length(Tperc) * length(Tstd) * length(fW);
imgpath = [{strcat('C:\dataset\data\',name,'\train\')} {strcat('C:\dataset\data\',name, '\val\')}];
imagefiles = [];
for i=1:length(imgpath)
   diagnosis = dir(imgpath{i}); 
   for x=3:length(diagnosis)
       curr_files = dir(strcat(imgpath{i},diagnosis(x).name,'\images\', '*_dlm_img'));
       for t=1:length(curr_files)
            imagefiles = [imagefiles; {strcat(imgpath{i},diagnosis(x).name,'\images\',curr_files(t).name)}]; 
       end
   end
end
%results = iterativeHmin('gnb.png', 20, 10, 5, 2);

indizes = 1:length(imagefiles);
totalcount = 0;
for a = Tsize
    for b = Tperc
        for c=Tstd
            for d=fW
                total_f1_score = [];
                total_AJI = [];
                cnt = 0;
                iwant = indizes(randperm(length(indizes),15));
                for i = iwant
                    img = imagefiles{i};
                    %convertDataset(img);
                    groundtruth = imread(strrep(strrep(strrep(strrep(img,'.jpg','.tif'),'Img_','Mask_'),'images','masks'),'_dlm_img','.tif'));
                    mask = strrep(img,'dlm_img','dlm_mask');
                    if length(unique(groundtruth(:)))>0
                        %prediction = iterativeHmin(img, a, b, c, d);
                        ARGraphs_Mex_NEU('1',img,mask,'prim',num2str(a),num2str(b),num2str(c),'-1','2');
                        ARGraphs_Mex_NEU('2','prim',mask,num2str(d),'segmented_cells','2');
                        prediction = dlmread('segmented_cells');
                        prediction = prediction(2:257,:);
                        [f1_score,AJI] = score(prediction,groundtruth);
                        total_f1_score = [total_f1_score;f1_score];
                        total_AJI = [total_AJI;AJI];
                        cnt = cnt + 1;
                        %figure(1),imshow(groundtruth>0,[]);figure(2),imshow(prediction,[]);
                        %disp('Parameters : ' + string(a) + ' ' + string(b) + ' ' + string(c) + ' ' + string(d))
                        %disp ('Image ' + imagefiles(i).name + ', F1-Score: ' + string(f1_score) + ', AJI-Score: ' + string(AJI))
                    end
                end
                if cnt > 0
                    disp(strcat('Mean score over ',num2str(cnt),' samples: F1-score=',num2str(mean(total_f1_score)),' +/- ', num2str(std(total_f1_score)),', AJI=',num2str(mean(total_AJI)),' +/- ',num2str(std(total_AJI))))
                    totalcount = totalcount + 1;
                    disp(strcat('Iteration ',num2str(totalcount),' from ',num2str(total_combinations)))
                    results = [results; a b c d mean(total_f1_score) std(total_f1_score) mean(total_AJI) std(total_AJI)];
                end
            end
        end
    end
end
foldername = strcat('results\results_' , name);
mkdir (foldername);
combined_results = (results(:,5) - results(:,6)) + (results(:,7) - results(:,8)) - abs(results(:,5)-results(:,6) - results(:,7) - results(:,8));
max_pos = find(combined_results==max(combined_results));
erg = results(max_pos,:); % Best parameter combination
if (plot_results)
    max_x = 400;
    FigH = figure('Position', get(0, 'Screensize'));plot(results(:,1),'g-','LineWidth',2);title('tArea');%xlim([0 max_x]);
    F = getframe(FigH);imwrite(F.cdata, strcat(foldername , '\TSize_fine.png'), 'png');
    close(FigH)
    FigH = figure('Position', get(0, 'Screensize')); plot(results(:,2),'m-','LineWidth',2);title('dSize');%xlim([0 max_x]);
    F = getframe(FigH);imwrite(F.cdata, strcat(foldername , '\TPerc_fine.png'), 'png');
    close(FigH)
    FigH = figure('Position', get(0, 'Screensize'));plot(results(:,3),'r-','LineWidth',2);title('gAngle');%xlim([0 max_x]);
    F = getframe(FigH);imwrite(F.cdata, strcat(foldername , '\TStd_fine.png'), 'png');
    close(FigH)
    FigH = figure('Position', get(0, 'Screensize'));plot(results(:,4),'b-','LineWidth',2);title('offset');%xlim([0 max_x]);
    F = getframe(FigH);imwrite(F.cdata, strcat(foldername , '\W_fine.png'), 'png');
    close(FigH);
    FigH = figure('Position', get(0, 'Screensize')),jbfill(1:length(results(:,5)),(results(:,5) - results(:,6))',(results(:,5) + results(:,6))',[0.9 0.4 0.4],[0.9 0.5 0.5],0,1);%xlim([0 max_x]);
    hold on; 
    plot([max_pos max_pos],[-0.2 1.2],'--k','LineWidth',2);ylim([-0.2 1.1]);
    plot(1:length(results(:,5)), results(:,5),'ko','LineWidth',1);title('F1 score');xlim([0 length(results(:,7))]);
    F = getframe(FigH);imwrite(F.cdata, strcat(foldername , '\F1_fine.png'), 'png');
    close(FigH)
    FigH = figure('Position', get(0, 'Screensize')),jbfill(1:length(results(:,7)),(results(:,7) - results(:,8))',(results(:,7) + results(:,8))',[0.4 0.9 0.4],[0.5 0.9 0.5],0,1);%xlim([0 max_x]);
    hold on; 
    plot([max_pos max_pos],[-0.2 1.2],'--k','LineWidth',2);ylim([-0.2 1.1]);
    plot(1:length(results(:,7)), results(:,7),'ko','LineWidth',1);title('AJI');xlim([0 length(results(:,7))]);
    F = getframe(FigH);imwrite(F.cdata, strcat(foldername , '\AJI_fine.png'), 'png');
    close(FigH)
    FigH = figure('Position', get(0, 'Screensize'))
    hold on; 
    plot([max_pos max_pos],[-1 1.2],'--k','LineWidth',2);ylim([-1 1.2]);
    plot([0 size(results,1)],[combined_results(max_pos) combined_results(max_pos)],'--k','LineWidth',2);ylim([-1 1.3]);xlim([0 length(results(:,7))]);
    plot(1:length(results(:,7)), combined_results(:),'ko','LineWidth',1);
    title('AJI + F1');
    F = getframe(FigH);imwrite(F.cdata, strcat(foldername , '\combined_fine.png'), 'png');
    close(FigH)
end

% Save results

save(strcat(foldername , '\results_fine.mat', 'results'));
save(strcat(foldername , '\combined_results_fine.mat', 'combined_results'));
save(strcat(foldername , '\results_chosen_fine.mat', 'erg'));
save(strcat(foldername , '\position_max_fine.mat', 'max_pos'));

