function allMets=evaluateTracking(seqmap,resDir,dataDir)
%% evaluate CLEAR MOT and other metrics
% concatenate ALL sequences and evaluate as one!
%
% SETUP:
%
% define directories for tracking results...
% resDir = fullfile('res','data',filesep);
% ... and the actual sequences
% dataDir = fullfile('..','data','2DMOT2015','train',filesep);
%
%

addpath(genpath('.'));


% read sequence map
seqmapFile=fullfile('seqmaps',seqmap);
allSeq = parseSequences(seqmapFile);

fprintf('Sequences: \n');
disp(allSeq')


% concat gtInfo
gtInfo=[];
gtInfo.X=[];
allFgt=zeros(1,length(allSeq));

% Find out the length of each sequence
% and concatenate ground truth
gtInfoSingle=[];
gtMat = [];
predMat = [];
seqCnt=0;
for s=allSeq
    seqCnt=seqCnt+1;
    seqName = char(s);
    seqFolder= [dataDir,seqName,filesep];
    
    assert(isdir(seqFolder),'Sequence folder %s missing',seqFolder);
    
    gtFile = fullfile(dataDir,seqName,'gt','gt.txt');
    gtI = convertTXTToStruct(gtFile,seqFolder);
    gtMat{seqCnt} = dlmread(gtFile);
    gtMat{seqCnt}(:,[1 2 7 8]) = gtMat{seqCnt}(:,[2 1 8 9]); 
    
    [Fgt,Ngt] = size(gtInfo.X);
    [FgtI,NgtI] = size(gtI.Xi);
    newFgt = Fgt+1:Fgt+FgtI;
    newNgt = Ngt+1:Ngt+NgtI;
    
    gtInfo.Xi(newFgt,newNgt) = gtI.Xi;
    gtInfo.Yi(newFgt,newNgt) = gtI.Yi;
    gtInfo.W(newFgt,newNgt) = gtI.W;
    gtInfo.H(newFgt,newNgt) = gtI.H;
    
    gtInfoSingle(seqCnt).wc=0;
    
    % fill in world coordinates if they exist
    if isfield(gtI,'Xgp') && isfield(gtI,'Ygp')
        gtInfo.Xgp(newFgt,newNgt) = gtI.Xgp;
        gtInfo.Ygp(newFgt,newNgt) = gtI.Ygp;
        gtInfoSingle(seqCnt).wc=1;
    end
    
    % check if bounding boxes available in solution
    imCoord=1;
    if all(gtI.Xi(find(gtI.Xi(:)))==-1)
        imCoord=0;
    end
    
    gtInfo.X=gtInfo.Xi;gtInfo.Y=gtInfo.Yi;
    if ~imCoord 
        gtInfo.X=gtInfo.Xgp;gtInfo.Y=gtInfo.Ygp; 
    end
    
    allFgt(seqCnt) = FgtI;
    
    gtInfoSingle(seqCnt).gtInfo=gtI;
    
    
    
    
end
gtInfo.frameNums=1:size(gtInfo.Xi,1);

allMets=[];

mcnt=1;


fprintf('Evaluating ... \n');


clear stInfo
stInfo.Xi=[];

evalMethod=1;

% flags for entire benchmark
% if one seq missing, evaluation impossible
eval2D=1;
eval3D=1;

seqCnt=0;

% iterate over each sequence
predMat = [];
for s=allSeq
    
    seqCnt=seqCnt+1;
    seqName = char(s);
    
    fprintf('\t... %s\n',seqName);
    
    % if a result is missing, we cannot evaluate this tracker
    resFile = fullfile(resDir,[seqName '.txt']);
    if ~exist(resFile,'file')
        fprintf('WARNING: result for %s not available!\n',seqName);
        eval2D=0;
        eval3D=0;
        continue;
    end
    
    % if MOT16, preprocess (clean)
    if cleanRequired(seqName)
        resFile = preprocessResult(resFile, seqName, dataDir);
    end

    
    predMat{seqCnt} = dlmread(resFile);
    predMat{seqCnt}(:,[1 2 7 8]) = predMat{seqCnt}(:,[2 1 8 9]);
    stI = convertTXTToStruct(resFile);
%     stI.Xi(find(stI.Xi(:)))=-1;
    % check if bounding boxes available in solution
    imCoord=1;
    if all(stI.Xi(find(stI.Xi(:)))==-1)
        imCoord=0;
    end
    
    worldCoordST=0; % state
    if isfield(stI,'Xgp') && isfield(stI,'Ygp')
        worldCoordST=1;
    end
    
    [FI,NI] = size(stI.Xi);
    
    
    % if stateInfo shorter, pad with zeros
    % GT and result must be equal length
    if FI<allFgt(seqCnt)
        missingFrames = FI+1:allFgt(seqCnt);
        stI.Xi(missingFrames,:)=0;
        stI.Yi(missingFrames,:)=0;
        stI.W(missingFrames,:)=0;
        stI.H(missingFrames,:)=0;
        stI.X(missingFrames,:)=0;
        stI.Y(missingFrames,:)=0;
        if worldCoordST
            stI.Xgp(missingFrames,:)=0; stI.Ygp(missingFrames,:)=0;
        end
        [FI,NI] = size(stI.Xi);
    % if stateInfo longer, crop
    elseif FI>allFgt(seqCnt)
        stI.Xi=stI.Xi(1:allFgt(seqCnt),:);
        stI.Yi=stI.Yi(1:allFgt(seqCnt),:);
        stI.W=stI.W(1:allFgt(seqCnt),:);
        stI.H=stI.H(1:allFgt(seqCnt),:);
        stI.X=stI.X(1:allFgt(seqCnt),:);
        stI.Y=stI.Y(1:allFgt(seqCnt),:);
        if worldCoordST
            stI.Xgp=stI.Xgp(1:allFgt(seqCnt),:); stI.Ygp=stI.Ygp(1:allFgt(seqCnt),:);
        end
        [FI,NI] = size(stI.Xi);
        
    end
    
    % get result for one sequence only
    [mets, mInf, addInf]=CLEAR_MOT_HUN(gtInfoSingle(seqCnt).gtInfo,stI);
    
    threshold = 0.5;
    world = ~imCoord;
    metsID = IDmeasures(predMat{seqCnt}, gtMat{seqCnt}, threshold, world);
    mets = [metsID.IDF1, metsID.IDP, metsID.IDR, mets];
   
    allMets(mcnt).mets2d(seqCnt).name=seqName;
    allMets(mcnt).mets2d(seqCnt).m=mets;
    allMets(mcnt).mets2d(seqCnt).additionalInfo=addInf;
    
    allMets(mcnt).mets3d(seqCnt).name=seqName;
    allMets(mcnt).mets3d(seqCnt).m=zeros(1,length(mets));
    
    if imCoord        
        fprintf('*** 2D (Bounding Box overlap) ***\n'); printMetrics(mets); fprintf('\n');
    else
        fprintf('*** Bounding boxes not available ***\n\n');
        eval2D=0;
    end
    
    % if world coordinates available, evaluate in 3D
    if  gtInfoSingle(seqCnt).wc &&  worldCoordST
        evopt.eval3d=1;evopt.td=1;
        [mets, mInf, addInf]=CLEAR_MOT_HUN(gtInfoSingle(seqCnt).gtInfo,stI,evopt);
        world = true;
        metsID = IDmeasures(predMat{seqCnt}, gtMat{seqCnt}, threshold, world);
        mets = [metsID.IDF1, metsID.IDP, metsID.IDR, mets];

        
        allMets(mcnt).mets3d(seqCnt).m=mets;
        allMets(mcnt).mets3d(seqCnt).additionalInfo=addInf;
                
        fprintf('*** 3D (in world coordinates) ***\n'); printMetrics(mets); fprintf('\n');            
    else
        eval3D=0;
    end
    
    
    [F,N] = size(stInfo.Xi);
    newF = F+1:F+FI;
    newN = N+1:N+NI;
    
    % concat result
    stInfo.Xi(newF,newN) = stI.Xi;
    stInfo.Yi(newF,newN) = stI.Yi;
    stInfo.W(newF,newN) = stI.W;
    stInfo.H(newF,newN) = stI.H;
    if isfield(stI,'Xgp') && isfield(stI,'Ygp')
        stInfo.Xgp(newF,newN) = stI.Xgp;stInfo.Ygp(newF,newN) = stI.Ygp;
    end
    stInfo.X=stInfo.Xi;stInfo.Y=stInfo.Yi;
    if ~imCoord 
        stInfo.X=stInfo.Xgp;stInfo.Y=stInfo.Ygp; 
    end
    
end
stInfo.frameNums=1:size(stInfo.Xi,1);

% Prepare data for ID measures for overall evaluation
gtMatAll = [];
predMatAll = [];
count = 0;
for k = 1:length(gtMat)
   gtMat{k}(:,2) = gtMat{k}(:,2) + count;
   predMat{k}(:,2) = predMat{k}(:,2) + count;
   count = count + max(gtMat{k}(:,2));
   gtMatAll = [gtMatAll; gtMat{k}]; 
   predMatAll = [predMatAll; predMat{k}];
end


if eval2D
    fprintf('\n');
    fprintf(' ********************* Your Benchmark Results (2D) ***********************\n');

%     [m2d, mInf]=CLEAR_MOT_HUN(gtInfo,stInfo);

    % It is faster to aggregate scores from all cameras than to re-evaluate
    MT = 0; PT = 0; ML = 0; FRA = 0;
    falsepositives = 0; missed = 0; idswitches = 0;
    Fgt = 0; iousum = 0; Ngt = 0; sumg = 0;
    Nc = 0;
    numGT = 0; numPRED = 0; IDTP = 0; IDFP = 0; IDFN = 0;
    
    for seqCnt = 1:length(allSeq)
        
        %     numGT = numGT + allMets(mcnt).mets2d(seqCnt).additionalInfo.IDmeasures.numGT;
        %     numPRED = numPRED + allMets(mcnt).mets2d(seqCnt).additionalInfo.IDmeasures.numPRED;
        %     IDTP = IDTP + allMets(mcnt).mets2d(seqCnt).additionalInfo.IDmeasures.IDTP;
        %     IDFN = IDFN + allMets(mcnt).mets2d(seqCnt).additionalInfo.IDmeasures.IDFN;
        %     IDFP = IDFP + allMets(mcnt).mets2d(seqCnt).additionalInfo.IDmeasures.IDFP;
        
        MT = MT + allMets(mcnt).mets2d(seqCnt).additionalInfo.MT;
        PT = PT + allMets(mcnt).mets2d(seqCnt).additionalInfo.PT;
        ML = ML + allMets(mcnt).mets2d(seqCnt).additionalInfo.ML;
        FRA = FRA + allMets(mcnt).mets2d(seqCnt).additionalInfo.FRA;
        Fgt = Fgt + allMets(mcnt).mets2d(seqCnt).additionalInfo.Fgt;
        Ngt = Ngt + allMets(mcnt).mets2d(seqCnt).additionalInfo.Ngt;
        Nc = Nc + sum(allMets(mcnt).mets2d(seqCnt).additionalInfo.c);
        sumg = sumg + sum(allMets(mcnt).mets2d(seqCnt).additionalInfo.g);
        falsepositives = falsepositives + sum(allMets(mcnt).mets2d(seqCnt).additionalInfo.fp);
        missed = missed + sum(allMets(mcnt).mets2d(seqCnt).additionalInfo.m);
        idswitches = idswitches + sum(allMets(mcnt).mets2d(seqCnt).additionalInfo.mme);
        ious = allMets(mcnt).mets2d(seqCnt).additionalInfo.ious;
        td = allMets(mcnt).mets2d(seqCnt).additionalInfo.td;
        iousum = iousum + sum(ious(ious>=td & ious<Inf));
    end
    
    FAR = falsepositives / Fgt;
    MOTP=iousum/Nc * 100; % avg ol
    if isnan(MOTP), MOTP=0; end % force to 0 if no matches found
    MOTAL=(1-(missed+falsepositives+log10(idswitches+1))/sumg)*100;
    MOTA=(1-(missed+falsepositives+idswitches)/sumg)*100;
    recall=Nc/sumg*100;
    precision=Nc/(falsepositives+Nc)*100;
    
    mets=[recall, precision, FAR, Ngt, MT, PT, ML, falsepositives, missed, idswitches, FRA, MOTA, MOTP, MOTAL];
    
    
    metsID = IDmeasures(predMatAll, gtMatAll, 0.5, false);
    m2d = [metsID.IDF1, metsID.IDP, metsID.IDR, mets];

    allMets.bmark2d=m2d;
    
    evalFile = fullfile(resDir, 'eval2D.txt');
    
    printMetrics(m2d);
    dlmwrite(evalFile,m2d);
end    

if eval3D
    fprintf('\n');
    fprintf(' ********************* Your Benchmark Results (3D) ***********************\n');

    evopt.eval3d=1;evopt.td=1;
       
    [m3d, mInf]=CLEAR_MOT_HUN(gtInfo,stInfo,evopt);
    metsID = IDmeasures(predMatAll, gtMatAll, evopt.td, evopt.eval3d);
    m3d = [metsID.IDF1, metsID.IDP, metsID.IDR, m3d];

    
    allMets.bmark3d=m3d;
    
    evalFile = fullfile(resDir, 'eval3D.txt');
    
    printMetrics(m3d);
    dlmwrite(evalFile,m3d);    
end
if ~eval2D && ~eval3D
    fprintf('ERROR: results cannot be evaluated\n');
end
