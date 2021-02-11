function erg = convertImage(file_name)
    %folder_name = 'C:\dataset\data\tiles_scaled_gold\test\Ganglioneuroma\images\';
    %file_name = 'Img_Ganglioneuroma_25.jpg';

    %dest_folder = 'Z:\ARGraphs\run\'

    img_name = strrep(strrep(file_name,'.jpg','_dlm_img'),'.png','_dlm_img'); %strcat(dest_folder,'img'); 
    mask_name = strrep(strrep(file_name,'.jpg','_dlm_mask'),'.png','_dlm_mask'); % strcat(dest_folder,'mask');
    
    img = imread(strcat(file_name));
    mask = ARG_initial_segmentation(img,50,5,250,0.5);

    fileID = fopen(img_name,'wt');
    nbytes = fprintf(fileID,'%d %d\n',256,256);
    fclose(fileID);

    fileID = fopen(mask_name,'wt');
    nbytes = fprintf(fileID,'%d %d\n',256,256);
    fclose(fileID);

    dlmwrite(img_name,img,'delimiter',' ','-append');
    dlmwrite(mask_name,mask,'delimiter',' ','-append');
end