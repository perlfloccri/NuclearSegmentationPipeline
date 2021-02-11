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