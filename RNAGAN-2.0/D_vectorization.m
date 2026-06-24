function V = D_vectorization...
    (Dnet,geneList,target,useGPU,reference)
%D_VECTORIZATION
%   Vectorize the given target using itself as refernce or a specificed
%   reference
%   author: Zhaozheng Hou (George)
%
% V = D_vectorization(Dnet,geneList,target,useGPU,reference)
% parameter:
%   - Dnet: discriminator network to use (network or the name of trained
%       network, such as "BDPP10" for "the discriminator for bulk-RNA seq
%       data with predefined pathways and using 10 references")
%   - geneList: list of genes (strings or numbers), leave blank and skip
%       the matching if the data already matched the gene list
%   - target: expression of targets, each column is one sample
%   - useGPU: (optional) whether using GPU for the processing or not, false
%       by default
%   - reference: (optional) expression of samples as reference, each column
%       is one sample
% output:
%   - V: 64D Vectors corresponding to the target samples , each column
%       is one sample

%% validate
if (isa(Dnet,"dlnetwork"))
    validateattributes(Dnet,{'dlnetwork'},{'scalar'});
else
    validateattributes(Dnet,{'string','char'},{'scalartext'});
    Dnet=load("core\"+string(Dnet)+".mat",string(Dnet));
    Dnet=struct2cell(Dnet);
    Dnet=Dnet{1};
end
n=Dnet.getLayer(Dnet.InputNames{1}).InputSize;

if (isempty(geneList))
    geneList=nan(n(1),1);
else
    validateattributes(geneList,{'numeric','char','string'},{'vector'});
end
validateattributes(target,{'numeric'},{'nonnegative','size',[numel(geneList),nan]});

if (nargin>=4)
    validateattributes(useGPU,{'numeric','logical'},{'binary'});
    if (isempty(useGPU))
        useGPU=false;
    end
else
    useGPU=false;
end

if (nargin>=5)
    validateattributes(reference,{'numeric'},{'nonnegative','size',[numel(geneList),nan]});
    validateattributes(size(reference,2),{'numeric'},{'>=',2});
else
    reference=target;
end

if (~Dnet.Initialized)
    warning("The network is not yet initialized. Initializing now...");
    Dnet=Dnet.initialize;
end

%% process
% match genes
if (~isnan(geneList))
    [netGeneID,listGeneID] = match_gene_list(geneList);
    temp=zeros(n(1),size(target,2));
    temp(netGeneID,:)=target(listGeneID,:);
    target=temp;
    temp=zeros(n(1),size(reference,2));
    temp(netGeneID,:)=reference(listGeneID,:);
    reference=temp;
end

validGene=any(target>0,2)&any(reference>0,2);
target(~validGene,:)=0;
reference(~validGene,:)=0;

% prepare references
ntrials=100; %number of trials for evaluation

Z=dlarray(nan(n(1),n(2),1,ntrials,'single'),'SSCB');
for ind=1:ntrials
    Z(:,:,1,ind)=reference(:,randi(size(reference,2),[n(2),1]));
end

if (useGPU)
    Z=gpuArray(Z);
end

% calculate V
if (mod(n(2),10)==1)
    n(2)=size(target,2);
    V=nan(64,n(2),'single');
    for ind=1:n(2)
        disp(num2str([ind,n(2)],"Processing %u out of %u targets..."));

        Z(:,1,1,:)=repmat(target(:,ind),[1,1,1,ntrials]);
        temp=predict(Dnet,Z,Outputs='swish_2');
        V(:,ind)=mean(temp,2);
    end
else
    V=predict(Dnet,Z,Outputs='swish_2');
    V=gather(V.extractdata);
end
end