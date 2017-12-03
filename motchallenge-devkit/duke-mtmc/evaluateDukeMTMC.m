function result = evaluateDukeMTMC(resMat, iou_threshold, world, testSet)

ROI = getROIs();

if strcmp(testSet,'easy')
    load('gt/testData.mat');
    gtMat = testData;
    testInterval = [263504:356648];
elseif strcmp(testSet,'hard')
    load('gt/testHardData.mat');
    gtMat = testHardData;
    testInterval = [227541:263503];
elseif strcmp(testSet,'trainval')
    load('gt/trainval.mat');
    gtMat = trainData;
    testInterval = [47720:227540]; % takes too long
elseif strcmp(testSet,'trainval_mini') % shorter version of trainval
    load('gt/trainval.mat');
    gtMat = trainData;
    testInterval = [127720:187540]; 
else
    
    fprintf('Unknown test set %s\n',testSet);
    return;
end



% Filter rows by frame interval
startTimes = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766];
for cam = 1:8
    gtMat(gtMat(:,1) == cam & ~ismember(gtMat(:,3) + startTimes(cam) - 1, testInterval),:) = [];
    resMat(resMat(:,1) == cam & ~ismember(resMat(:,3) + startTimes(cam) - 1, testInterval),:) = [];
end

% Filter rows by feet position within ROI
feetpos = [ resMat(:,4) + 0.5*resMat(:,6), resMat(:,5) + resMat(:,7)];
keep = false(size(resMat,1),1);
for cam = 1:8
    camFilter = resMat(:,1) == cam;
    keep(camFilter & inpolygon(feetpos(:,1),feetpos(:,2), ROI{cam}(:,1),ROI{cam}(:,2))) = true;
end

resMat = resMat(keep,:);
 
if isempty(resMat)
   result = [];
   return
end

% Single-Cam
for camera = 1:8
    fprintf('Processing camera %d...\n',camera);
    resMatSingle = resMat(resMat(:,1)==camera, 2:7);
    gtMatSingle = gtMat(gtMat(:,1)==camera, 2:7);
    measures = IDmeasures(resMatSingle, gtMatSingle, iou_threshold, world);
    result{camera}.IDmeasures = measures;
    result{camera}.description = sprintf('Cam_%d',camera);
    result{camera}.allMets = evaluateTracking(result{camera}.description, gtMatSingle, resMatSingle);
    result{camera}.allMets.mets2d.m = [measures.IDF1, measures.IDP, measures.IDR, result{camera}.allMets.mets2d.m];

    
end
fprintf('\n');


% Multi-Cam

% Convert data format to:
% ID, frame, left, top, width, height, worldX, worldY
SHIFT_CONSTANT = 100000000;

gtMatMulti  = gtMat(:,2:7);
resMatMulti = resMat(:,2:7);
gtMatMulti(:,2) = gtMat(:,3) + gtMat(:,1)*SHIFT_CONSTANT; % frame + cam*1000000 for frame uniqueness
resMatMulti(:,2) = resMat(:,3) + resMat(:,1)*SHIFT_CONSTANT; 
result{10}.IDmeasures = IDmeasures(resMatMulti, gtMatMulti, iou_threshold, world);
result{10}.description = 'Multi-cam';

% AllCameraSingle (MC Upper bound) 
gtMatSingleAll = gtMat(:,2:7);
resMatSingleAll = resMat(:,2:7);

gtMatSingleAll(:,1) = gtMatSingleAll(:,1) + gtMat(:,1)*SHIFT_CONSTANT; % ID + cam*1000000 for ID uniqueness
resMatSingleAll(:,1) = resMatSingleAll(:,1) + resMat(:,1)*SHIFT_CONSTANT;

for cam = 1:8 % frame uniqueness
    gtMatSingleAll(gtMat(:,1)==cam,2) = gtMatSingleAll(gtMat(:,1)==cam,2) + (cam-1) * numel(testInterval);
    resMatSingleAll(resMat(:,1)==cam,2)  = resMatSingleAll(resMat(:,1)==cam,2) + (cam-1) * numel(testInterval);
end


result{9}.description = 'Single-all';
if false
    measures = IDmeasures(resMatSingleAll, gtMatSingleAll, iou_threshold, world);
    result{9}.IDmeasurs = measures;
    result{9}.allMets = evaluateTracking(result{9}.description, gtMatSingleAll, resMatSingleAll);
else
    % It is faster to aggregate scores from all cameras than to re-evaluate
    MT = 0; PT = 0; ML = 0; FRA = 0;
    falsepositives = 0; missed = 0; idswitches = 0;
    Fgt = 0; iousum = 0; Ngt = 0; sumg = 0;
    Nc = 0;
    numGT = 0; numPRED = 0; IDTP = 0; IDFP = 0; IDFN = 0;
    
    for cam = 1:8
        
        numGT = numGT + result{cam}.IDmeasures.numGT;
        numPRED = numPRED + result{cam}.IDmeasures.numPRED;
        IDTP = IDTP + result{cam}.IDmeasures.IDTP;
        IDFN = IDFN + result{cam}.IDmeasures.IDFN;
        IDFP = IDFP + result{cam}.IDmeasures.IDFP;
        
        MT = MT + result{cam}.allMets.mets2d.additionalInfo.MT;
        PT = PT + result{cam}.allMets.mets2d.additionalInfo.PT;
        ML = ML + result{cam}.allMets.mets2d.additionalInfo.ML;
        FRA = FRA + result{cam}.allMets.mets2d.additionalInfo.FRA;
        Fgt = Fgt + result{cam}.allMets.mets2d.additionalInfo.Fgt;
        Ngt = Ngt + result{cam}.allMets.mets2d.additionalInfo.Ngt;
        Nc = Nc + sum(result{cam}.allMets.mets2d.additionalInfo.c);
        sumg = sumg + sum(result{cam}.allMets.mets2d.additionalInfo.g);
        falsepositives = falsepositives + sum(result{cam}.allMets.mets2d.additionalInfo.fp);
        missed = missed + sum(result{cam}.allMets.mets2d.additionalInfo.m);
        idswitches = idswitches + sum(result{cam}.allMets.mets2d.additionalInfo.mme);
        ious = result{cam}.allMets.mets2d.additionalInfo.ious;
        td = result{cam}.allMets.mets2d.additionalInfo.td;
        iousum = iousum + sum(ious(ious>=td & ious<Inf));
    end
    
    IDPrecision = IDTP / (IDTP + IDFP);
    IDRecall = IDTP / (IDTP + IDFN);
    IDF1 = 2*IDTP/(numGT + numPRED);

    measures.IDP = IDPrecision * 100;
    measures.IDR = IDRecall * 100;
    measures.IDF1 = IDF1 * 100;
    measures.numGT = numGT;
    measures.numPRED = numPRED;
    measures.IDTP = IDTP;
    measures.IDFP = IDFP;
    measures.IDFN = IDFN;
    result{9}.IDmeasures = measures;
    
    FAR = falsepositives / Fgt;
    MOTP=iousum/Nc * 100; % avg ol
    MOTAL=(1-(missed+falsepositives+log10(idswitches+1))/sumg)*100;
    MOTA=(1-(missed+falsepositives+idswitches)/sumg)*100;
    recall=Nc/sumg*100;
    precision=Nc/(falsepositives+Nc)*100;
    
    metrics=[recall, precision, FAR, Ngt, MT, PT, ML, falsepositives, missed, idswitches, FRA, MOTA, MOTP, MOTAL];
    result{9}.allMets.mets2d.m = metrics;
end

result{9}.allMets.mets2d.m = [measures.IDF1, measures.IDP, measures.IDR, result{9}.allMets.mets2d.m];

