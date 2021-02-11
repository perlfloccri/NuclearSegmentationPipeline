%name = 'tiles_scaled_silver_revision';
%imgpath = [{strcat('C:\dataset\data\',name,'\train\')} {strcat('C:\dataset\data\',name, '\val\')} {strcat('C:\dataset\data\',name, '\test\')}];
name = 'convertedArtificialImages_gold';
imgpath = [{strcat('C:\dataset\data\',name,'\scaled\train\')} {strcat('C:\dataset\data\',name, '\scaled\val\')}];
imagefiles = [];
for i=1:length(imgpath)
   diagnosis = dir(imgpath{i}); 
   for x=3:length(diagnosis)
       %curr_files = dir(strcat(imgpath{i},diagnosis(x).name,'\images\', '*.jpg'));
       curr_files = dir(strcat(imgpath{i},diagnosis(x).name,'\images\', '*.png'));
       
       for t=1:length(curr_files)
           disp(strcat('Currently processing: ',curr_files(t).name)) 
           convertImage(strcat(imgpath{i},diagnosis(x).name,'\images\',curr_files(t).name));
       end
   end
end