%% Compile

mex devkit/utils/assignmentoptimal.c -outdir devkit/utils
mex devkit/utils/clearMOTMex.cpp -outdir devkit/utils
mex devkit/utils/costBlockMex.cpp -outdir devkit/utils COMPFLAGS="/openmp $COMPFLAGS"


%% Fetching data

% Fetching data
if ~exist('gt/trainval.mat','file')
    fprintf('Downloading ground truth...\n');
    url = 'http://vision.cs.duke.edu/DukeMTMC/data/ground_truth/trainval.mat';
    if ~exist('gt','dir'), mkdir('gt'); end
    filename = 'gt/trainval.mat';
    outfilename = websave(filename,url);
end
if ~exist('res/baseline.txt','file')
    fprintf('Downloading baseline tracker output...\n');
    url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/tracker_output.zip';
    if ~exist('res','dir'), mkdir('res'); end
    filename = 'res/tracker_output.zip';
    outfilename = websave(filename,url);
    unzip(outfilename,'res');
    delete(filename);
end

%% Evaluation
addpath(genpath('devkit'));

trackerOutput = dlmread('res/baseline.txt');

world = false; % Image plane
iou_threshold = 0.5;
testSets = {'trainval'}; 
% testSets = {'trainval_mini'}; 


% Evaluate
for i = 1:length(testSets)
    testSet = testSets{i};
    results{i} = evaluateDukeMTMC(trackerOutput, iou_threshold, world, testSet);

end

%% Display

% Print
for i = 1:length(testSets)
    
    result = results{i};
    fprintf('\n-------Results-------\n');
    fprintf('Test set: %s\n', testSets{i});
    
    if isempty(result)
        fprintf('No input data found\n');
        continue;
    end
    
    % Single cameras all
    fprintf('%s\n',result{9}.description);
    printMetrics(result{9}.allMets.mets2d.m); 
    
    % Multi-cam
    k = 10;    
    fprintf(result{k}.description);
    fprintf('\tIDF1: %.2f', result{k}.IDmeasures.IDF1);
    fprintf('\tIDP: %.2f', result{k}.IDmeasures.IDP);
    fprintf('\tIDR: %.2f\n', result{k}.IDmeasures.IDR);
    
    % All individual cameras
    fprintf('\n'); 
    for k = 1:8
       fprintf('%s\n',result{k}.description);
       printMetrics(result{k}.allMets.mets2d.m); 
    end

end