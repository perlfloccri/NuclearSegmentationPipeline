function [allMarkers, allMarkersOrg] = findMarkers(map, binMask, tArea, dSize, initH)
    [dx, dy]       = size (map);
    allMarkers     = zeros (dx,dy);
    allMarkersOrg  = zeros (dx,dy);
    
    h = initH;
    while 1
        if max(max(map)) > 0
            hmap = imhmin (map, h);
            currMarkers = imregionalmin (hmap);
            currMarkers = maskBasedOnArea (currMarkers, tArea);
            currMarkers = maskBasedOnMap (currMarkers, binMask);
            currMarkers = eliminateOverlappingAreas (allMarkers, currMarkers, dx, dy);
            [allMarkers, allMarkersOrg, inserted] = insertNewMarkers (allMarkers, allMarkersOrg, currMarkers, dSize);
            if inserted == 0
                return;
            else
                h = h + 1;    
            end
        else
            return;
        end
    end
end

function markers = maskBasedOnArea (markers, tArea)
    if tArea ~= -1
        markers = bwareaopen (markers, tArea);
    end
end

function markers = maskBasedOnMap (markers, binMask)
    [L n] = bwlabel (markers);
    for i=1:n
        t = L == i;
        if (sum (sum (binMask & t)) / sum(sum(t))) < 1
            markers (L == i) = 0;
        end
    end
end

function curr = eliminateOverlappingAreas (prev, curr, dx, dy)
    cc = bwlabeln (curr);
    maxCC = max (max (cc));
    overlappingCC = zeros (maxCC,1);
    for i=1:dx
        for j=1:dy
            if cc (i,j) > 0
                if prev (i,j) > 0
                    overlappingCC (cc (i,j)) = overlappingCC (cc (i,j)) + 1;
                end
            end
        end
    end
    
    eliminateCC = zeros (maxCC,1);
    for i=1:maxCC
        if overlappingCC (i) > 0
            eliminateCC (i) = 1;
        end
    end
    for i=1:dx
        for j=1:dy
            if cc (i,j) > 0 && eliminateCC (cc (i,j)) == 1
                curr (i,j) = 0;
            end
        end
    end
end

function [allMarkers, allMarkersOrg, inserted] = insertNewMarkers (allMarkers, allMarkersOrg, currMarkers, dSize)
    maxNo = max (max (allMarkers));
    cc = bwlabeln (currMarkers);
    if dSize ~= -1
        se = strel ('disk', dSize);
        cc2 = imdilate (cc, se);
    end
    
    inserted = max (max (cc));

    cc2(cc2>0) = cc2 (cc2>0) + maxNo;
    allMarkers (cc2>0) = cc2 (cc2>0);
    cc(cc>0) = cc (cc>0) + maxNo;
    allMarkersOrg (cc>0) = cc (cc>0);
end

