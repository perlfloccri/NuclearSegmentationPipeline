addpath('C:\Users\flo\Desktop\IterativeHMinima-master');
addpath('C:\Users\flo\Desktop\Classical_algorithms\natsort');

foldername = "results_tiles_scaled_silver";
folder_inference = "C:\dataset\data\tiles_scaled_silver_revision\test\";

actual_folders = dir(folder_inference);
image_files = [];
for x=3:length(actual_folders)
    curr_files = dir(strcat(folder_inference,actual_folders(x).name + "\images\", '*.jpg'));
    curr_files = natsortfiles({curr_files.name});
    for t=1:length(curr_files)
        image_files = [image_files;strcat(folder_inference,actual_folders(x).name, "\images\", curr_files{t})];
    end
end

load("iterativehmin\" + foldername + "\results_chosen_fine.mat");
a=erg(1);
b=erg(2);
c=erg(3);
d=erg(4);

prediction_results = cell(length(image_files),1);
for i=1:length(image_files)
    if i==200
        e=1
    end
    prediction = iterativeHmin(image_files(i), a, b, c, d);
    prediction_results{i} = compute_final_mask(prediction);
    disp ("Running crop " + string(i) + " from " + string(length(image_files)))
end
save(foldername + "_predictions.mat","prediction_results");

function mask_new = compute_final_mask(mask)
    %figure(1);
    %subplot(1,2,1);imshow(mask,[]);
    mask_new = zeros(size(mask,1),size(mask,2));
    labels = unique(mask);
    for i=1:length(labels)
        if ((labels(i) ~= 0) && (sum(sum(mask==labels(i))) < 11000))
            mask_new = mask_new + (imerode(mask==labels(i), strel('disk',1)) > 0);
        end
    end
    mask_new = uint8(mask_new);
    %subplot(1,2,2);imshow(mask_new,[]);
end
