%segmRes = objectOrientedSegmentation ('hepg2f_1.jpg', 5, 20, 0.1, 400);
addpath('z:\ARGraphs\x64\Release');
addpath('z:\ARGraphs');
addpath('Z:\ARGraphs\arggraphs_segmentation\natsort');
plot_results = 1;
name = 'tiles_scaled_gold';
% Non-scaled samples
% 20x :
%results = ['tarea' 'dSize' 'gAngle' 'offset' 'F1 score' 'AJI'];

folder_name = 'results_tiles_scaled_gold';
folder_inference = 'C:\dataset\data\tiles_scaled_gold_revision\test\';

actual_folders = dir(folder_inference);
image_files = [];
for x=3:length(actual_folders)
    curr_files = dir(strcat(folder_inference,actual_folders(x).name,'\images\', '*_dlm_img'));
    curr_files = natsortfiles({curr_files.name});
    for t=1:length(curr_files)
        image_files = [image_files; {strcat(folder_inference,actual_folders(x).name,'\images\',curr_files(t))}]; 
    end
end

load(strcat('Z:\ARGraphs\results\',folder_name,'\results_chosen_fine.mat'));
a=erg(1);
b=erg(2);
c=erg(3);
d=erg(4);

prediction_results = cell(length(image_files),1);
for i=1:length(image_files)
    img = image_files{i};
    mask = strrep(img,'dlm_img','dlm_mask');
    ARGraphs_Mex_NEU('1',img,mask,'prim',num2str(a),num2str(b),num2str(c),'-1','2');
    ARGraphs_Mex_NEU('2','prim',mask,num2str(d),'segmented_cells','2');
    prediction = dlmread('segmented_cells');
    prediction = prediction(2:257,:);
    prediction_results{i} = compute_final_mask(prediction);
    disp (strcat('Running crop ' , num2str(i) , ' from ' , num2str(length(image_files))))
end
save(strcat('arg_',folder_name,'_predictions_updated'),'prediction_results');


