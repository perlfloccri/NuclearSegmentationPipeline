if strfind(computer(),'64')
    defs = '-DA64BITS '; % for 64bit machines - define pointer type
else
    defs = '';
end
if verLessThan('matlab','7.3')    
    defs = [defs, '-DmwIndex=int -DmwSize=size_t '];
end

cmd = sprintf('mex -O -largeArrayDims %s ../imageProcessing.c ../linear.c matrix.c ../util.c', defs)
eval(cmd);
cmd = sprintf('mex -O -largeArrayDims %s cellCandidates.c markerDefinition.c primitiveOperations.c sobelPrimitives.c sobelTasks.c watershed.c main.c', defs)
eval(cmd);

clear cmd mj mn v di defs


