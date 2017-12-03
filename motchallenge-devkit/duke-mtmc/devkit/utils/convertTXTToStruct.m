function stInfo=convertTXTToStruct(allData, range)
% read CSV file and convert to Matlab struct format
% Modified by Ergys Ristani (ristani@cs.duke.edu)
% 

% load text file
numCols=size(allData,2);
numLines=size(allData,1);



% quickly check format
assert(numCols>=6,'FORMAT ERROR: Each line must have at least 6 values');
assert(all(allData(:,1)>0),'FORMAT ERROR: Frame numbers must be positive.');
assert(all(allData(:,2)>0),'FORMAT ERROR: IDs must be positive.');

imCoord=1;

% Empty state, still evaluate
if numLines == 0
    stInfo.W(range(end),1) = 0;
    stInfo.H(range(end),1) = 0;
    stInfo.Xi(range(end),1) = 0;
    stInfo.Yi(range(end),1) = 0;
end

% go through all lines
for l=1:numLines
   
    if ~mod(l,10000), fprintf('.'); end % print each 10,000th line
    lineData=allData(l,:);

    % ignore 0-marked GT
    if length(lineData) > 6 && ~lineData(7)
        continue;
    end    
    
    id = lineData(1);   % frame number
    fr = lineData(2);   % target id
    
    % bounding box    
    stInfo.W(fr,id) = lineData(5);
    stInfo.H(fr,id) = lineData(6);
    stInfo.Xi(fr,id) = lineData(3) + stInfo.W(fr,id)/2;
    stInfo.Yi(fr,id) = lineData(4) + stInfo.H(fr,id);
end
fprintf('\n');

% append empty frames?
if nargin>1
    
    if range(end) > size(stInfo.W,1)
        missingFrames = size(stInfo.W,1)+1:range(end);
        stInfo.Xi(missingFrames,:)=0;
        stInfo.Yi(missingFrames,:)=0;
        stInfo.W(missingFrames,:)=0;
        stInfo.H(missingFrames,:)=0;
       
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
nzc=~~sum(stInfo.Xi);

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

% do sparse matrices
if isfield(stInfo,'X')
    stInfo.X=sparse(stInfo.X);
    stInfo.Y=sparse(stInfo.Y);
end
if isfield(stInfo,'Xgp')
    stInfo.Xgp=sparse(stInfo.Xgp);    
    stInfo.Ygp=sparse(stInfo.Ygp);    
end
if isfield(stInfo,'Xi')
    stInfo.Xi=sparse(stInfo.Xi);
    stInfo.Yi=sparse(stInfo.Yi);
    stInfo.W=sparse(stInfo.W);
    stInfo.H=sparse(stInfo.H);
end