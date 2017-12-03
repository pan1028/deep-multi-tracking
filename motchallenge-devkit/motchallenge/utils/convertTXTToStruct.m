function stInfo=convertTXTToStruct(txtFile,seqFolder,loadFull, loadDet)
% read CSV file and convert to Matlab struct format

% fprintf('Importing data from %s\n',txtFile);

if nargin<3, loadFull=false; end
if nargin<4, loadDet=false; end

% if file empty, return empty (null) solution
tf = dir(txtFile);
F=0;
emptySol = zeros(F,0);
stInfo.X=emptySol; stInfo.Y=emptySol; stInfo.Xi=emptySol; stInfo.Yi=emptySol;
stInfo.W=emptySol; stInfo.H=emptySol;
if loadDet, stInfo.S=emptySol; end
if tf.bytes == 0	
    return;
end

% load text file
allData = dlmread(txtFile);
numCols=size(allData,2);
numLines=size(allData,1);

% quickly check format
assert(numCols>=7,'FORMAT ERROR: Each line must have at least 6 values');
% assert(all(allData(:,1)>0),'FORMAT ERROR: Frame numbers must be positive.');
% assert(all(allData(:,2)>0),'FORMAT ERROR: IDs must be positive.');

IDMap = unique(allData(:,2))';
nIDs = length(IDMap);
% fprintf('%d unique targets IDs discovered\n',nIDs);
% pause

% do we have bbox coordinates?
imCoord=0;
if any(any(allData(:,3:6)~= -1))
    imCoord=1;
end

% do we have world coordinates?
worldCoord=0;

% ... we do if they exist and are not -1 (default)
if size(allData,2)>=9
    if any(any(allData(:,[8 9])~= -1))
        worldCoord=1; 
    end
end

% at least one representation must be defined
assert(imCoord || worldCoord, ...
    'FORMAT ERROR: Neither bounding boxes nor world coordinates defined.');

% are we dealing with MOT16/17 ground truth?
MOT16GT=false;
if numCols==9 && nargin>1 && (cleanRequired(seqFolder))
    MOT16GT=true;
end

% go through all lines
for l=1:numLines
    lineData=allData(l,:);
    
    if ~loadFull
        % ignore 0-marked GT
        if ~lineData(7), continue; end

        % ignore non-pedestrians for MOT16
        if MOT16GT && lineData(8)~= 1, continue; end
    end
    
    fr = lineData(1);   % frame number
    id = lineData(2);   % target id

    % map id to 1..nIDs
    id = find(IDMap==id);
    if loadDet % don't care about ID for dets, just append
        if size(stInfo.W,1)<fr
            id=1;
        else
            id=find(stInfo.W(fr,:),1,'last')+1;
        end
    end
    
    %%%% sanity checks
    % ignore non-positive frames and IDs
    if fr<1, continue; end
    
    % ignore too large frame numbers
    if fr>1e4, continue; end
    
    
    
    
    % bounding box    
    stInfo.W(fr,id) = lineData(5);
    stInfo.H(fr,id) = lineData(6);
    stInfo.Xi(fr,id) = lineData(3) + stInfo.W(fr,id)/2;
    stInfo.Yi(fr,id) = lineData(4) + stInfo.H(fr,id);
		
		% score for detections
		if loadDet, stInfo.S(fr,id) = lineData(7); end
    
    % consider 3D coordinates
    if worldCoord
        stInfo.Xgp(fr,id) = lineData(8);
        stInfo.Ygp(fr,id) = lineData(9);
        
        % position should not be exactly 0
        if ~stInfo.Xgp(fr,id)
            stInfo.Xgp(fr,id)=stInfo.Xgp(fr,id)+0.0001;
        end
        if ~stInfo.Ygp(fr,id)
            stInfo.Ygp(fr,id)=stInfo.Ygp(fr,id)+0.0001;
        end
    end
		if MOT16GT
			stInfo.Flag(fr,id) = lineData(7);
			stInfo.Class(fr,id) = lineData(8);
			stInfo.Vis(fr,id) = lineData(9);
		end
end

% append empty frames?
if nargin>1
    imgFolders = dir(fullfile(seqFolder,filesep,'img*'));
    if (isempty(imgFolders))
        imgFolders = dir(fullfile(seqFolder,filesep,'lef*'));
    end
    imgFolder = fullfile(seqFolder,imgFolders(1).name,filesep);
    imgExt=getImgExt(imgFolder);

    imgMask=[imgFolder,'*' imgExt];
    dirImages = dir(imgMask);
    Fgt=length(dirImages);
    F=size(stInfo.W,1);
    % if stateInfo shorter, pad with zeros
    if F<Fgt
        missingFrames = F+1:Fgt;
        stInfo.Xi(missingFrames,:)=0;
        stInfo.Yi(missingFrames,:)=0;
        stInfo.W(missingFrames,:)=0;
        stInfo.H(missingFrames,:)=0;
        if worldCoord
            stInfo.Xgp(missingFrames,:)=0;
            stInfo.Ygp(missingFrames,:)=0;
        end
				if MOT16GT
					stInfo.Flag(missingFrames,:)=0;
					stInfo.Class(missingFrames,:)=1;
					stInfo.Vis(missingFrames,:)=1;
				end
				if loadDet,	stInfo.S(missingFrames,:)=0; end
    end
end

% set X,Y
stInfo.X=stInfo.Xi;stInfo.Y=stInfo.Yi;
if ~imCoord 
    stInfo.X=stInfo.Xgp;stInfo.Y=stInfo.Ygp; 
    
    % reset image coordinates to -1
    te = find(stInfo.X(:));
    stInfo.W(te)=-1;stInfo.H(te)=-1;
    stInfo.Xi(te)=-1;stInfo.Yi(te)=-1;
end


% remove empty target IDs
nzc=~~sum(stInfo.Xi,1);

if isfield(stInfo,'X')
    stInfo.X=stInfo.X(:,nzc);
    stInfo.Y=stInfo.Y(:,nzc);
end

% nzc
% stInfo.Xgp'
if isfield(stInfo,'Xgp')
    stInfo.Xgp=stInfo.Xgp(:,nzc);
    stInfo.Ygp=stInfo.Ygp(:,nzc); 
end

if isfield(stInfo,'Xi')
    stInfo.Xi=stInfo.Xi(:,nzc);
    stInfo.Yi=stInfo.Yi(:,nzc); 
    stInfo.W=stInfo.W(:,nzc);
    stInfo.H=stInfo.H(:,nzc);
end

if isfield(stInfo,'Flag')
	stInfo.Flag=stInfo.Flag(:,nzc);
	stInfo.Class=stInfo.Class(:,nzc);
	stInfo.Vis=stInfo.Vis(:,nzc);
end	

if isfield(stInfo,'S')
	stInfo.S = stInfo.S(:,nzc);
end