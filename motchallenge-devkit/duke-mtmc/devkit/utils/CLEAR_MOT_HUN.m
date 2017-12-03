function [metrics metricsInfo additionalInfo]=CLEAR_MOT_HUN(gtInfo,stateInfo,options)
% compute CLEAR MOT and other metrics
%
% metrics contains the following
% [1]   recall	- recall = percentage of detected targets
% [2]   precision	- precision = percentage of correctly detected targets
% [3]   FAR		- number of false alarms per frame
% [4]   GT        - number of ground truth trajectories
% [5-7] MT, PT, ML	- number of mostly tracked, partially tracked and mostly lost trajectories
% [8]   falsepositives- number of false positives (FP)
% [9]   missed        - number of missed targets (FN)
% [10]  idswitches	- number of id switches     (IDs)
% [11]  FRA       - number of fragmentations
% [12]  MOTA	- Multi-object tracking accuracy in [0,100]
% [13]  MOTP	- Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)
% [14]  MOTAL	- Multi-object tracking accuracy in [0,100] with log10(idswitches)
%
% 
% (C) Anton Milan, 2012-2014


% default options: 2D
if nargin<3
    options.eval3d=0;   % only bounding box overlap
    options.td=.5;      % threshold 50%
end

if ~isfield(options,'td')
    if options.eval3d
        options.td=1000;
    else
        options.td=0.5;
    end
end

td=options.td;

% if X,Y not existent, assume 2D
if ~isfield(gtInfo,'X'), gtInfo.X=gtInfo.Xi; end
if ~isfield(gtInfo,'Y'), gtInfo.Y=gtInfo.Yi; end
if ~isfield(stateInfo,'X'), stateInfo.X=stateInfo.Xi; end
if ~isfield(stateInfo,'Y'), stateInfo.Y=stateInfo.Yi; end

gtInd=~~gtInfo.X;
stInd=~~stateInfo.X;

[Fgt, Ngt]=size(gtInfo.X);
[F, N]=size(stateInfo.X);

% if stateInfo shorter, pad with zeros
if F<Fgt
    missingFrames = F+1:Fgt;
    stateInfo.Xi(missingFrames,:)=0;
    stateInfo.Yi(missingFrames,:)=0;
    stateInfo.W(missingFrames,:)=0;
    stateInfo.H(missingFrames,:)=0;
end



metricsInfo.names.long = {'Recall','Precision','False Alarm Rate', ...
    'GT Tracks','Mostly Tracked','Partially Tracked','Mostly Lost', ...
    'False Positives', 'False Negatives', 'ID Switches', 'Fragmentations', ...
    'MOTA','MOTP', 'MOTA Log'};

metricsInfo.names.short = {'Rcll','Prcn','FAR', ...
    'GT','MT','PT','ML', ...
    'FP', 'FN', 'IDs', 'FM', ...
    'MOTA','MOTP', 'MOTAL'};

metricsInfo.widths.long = [6 9 16 9 14 17 11 15 15 11 14 5 5 8];
metricsInfo.widths.short = [5 5 5 3 3 3 3 4 4 3 3 5 5 5];

metricsInfo.format.long = {'.1f','.1f','.2f', ...
    'i','i','i','i', ...
    'i','i','i','i', ...
    '.1f','.1f','.1f'};

metricsInfo.format.short=metricsInfo.format.long;


metrics=zeros(1,14);
metrics(9)=numel(find(gtInd));  % False Negatives (missed)
metrics(7)=Ngt;                 % Mostly Lost
metrics(4)=Ngt;                 % GT Trajectories

additionalInfo=[];
% nothing to be done, if state is empty
% if ~N, return; end

% Slower Matlab loop
% [mme, c, fp, m, g, d, ious, alltracked, allfalsepos, M] = clearMOTLoopMatlab(gtInfo, stateInfo, options);

% Faster CLEAR MOT computation
fields = fieldnames(gtInfo);
for i = 1:numel(fields)
    gtInfo.(fields{i}) = full(gtInfo.(fields{i}));
end
fields = fieldnames(stateInfo);
for i = 1:numel(fields)
    stateInfo.(fields{i}) = full(stateInfo.(fields{i}));
end
[mme, c, fp, m, g, d, ious, alltracked, allfalsepos, M] = clearMOTMex(gtInfo, stateInfo, td, options.eval3d);

fprintf('\n');

missed=sum(m);
falsepositives=sum(fp);
idswitches=sum(mme);

if options.eval3d
    MOTP=(1-sum(sum(d))/sum(c)/td) * 100; % avg distance to [0,100]
else
    MOTP=sum(ious(ious>=td & ious<Inf))/sum(c) * 100; % avg ol
end
if isnan(MOTP), MOTP=0; end % force to 0 if no matches found

MOTAL=(1-((sum(m)+sum(fp)+log10(sum(mme)+1))/sum(g)))*100;
MOTA=(1-((sum(m)+sum(fp)+(sum(mme)))/sum(g)))*100;
recall=sum(c)/sum(g)*100;
precision=sum(c)/(sum(fp)+sum(c))*100;
if isnan(precision), precision=0; end % force to 0 if no matches found
FAR=sum(fp)/Fgt;
 

%% MT PT ML
MTstatsa=zeros(1,Ngt);
for i=1:Ngt
    gtframes=find(gtInd(:,i));
    gtlength=length(gtframes);
    gttotallength=numel(find(gtInd(:,i)));
    trlengtha=numel(find(alltracked(gtframes,i)>0));
    if gtlength/gttotallength >= 0.8 && trlengtha/gttotallength < 0.2
        MTstatsa(i)=3;
    elseif F>=find(gtInd(:,i),1,'last') && trlengtha/gttotallength <= 0.8
        MTstatsa(i)=2;
    elseif trlengtha/gttotallength >= 0.8
        MTstatsa(i)=1;
    end
end
% MTstatsa
MT=numel(find(MTstatsa==1));PT=numel(find(MTstatsa==2));ML=numel(find(MTstatsa==3));

%% fragments
fr=zeros(1,Ngt);
for i=1:Ngt
    b=alltracked(find(alltracked(:,i),1,'first'):find(alltracked(:,i),1,'last'),i);
    b(~~b)=1;
    fr(i)=numel(find(diff(b)==-1));
end
FRA=sum(fr);

assert(Ngt==MT+PT+ML,'Hmm... Not all tracks classified correctly.');
metrics=[recall, precision, FAR, Ngt, MT, PT, ML, falsepositives, missed, idswitches, FRA, MOTA, MOTP, MOTAL];

additionalInfo.alltracked=alltracked;
additionalInfo.allfalsepos=allfalsepos;
additionalInfo.m = m;
additionalInfo.fp = fp;
additionalInfo.mme = mme;
additionalInfo.g = g;
additionalInfo.c = c;
additionalInfo.Fgt = Fgt;
additionalInfo.Ngt = Ngt;
additionalInfo.ious = ious;
additionalInfo.td = td;
additionalInfo.MT = MT;
additionalInfo.PT = PT;
additionalInfo.ML = ML;
additionalInfo.FRA = FRA;
end 



