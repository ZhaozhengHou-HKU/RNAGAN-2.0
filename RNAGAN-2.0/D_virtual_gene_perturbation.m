function [perturbatedSample,DE_log2FC,Pvalue] = D_virtual_gene_perturbation...
    (Dnet,geneList,samples2perturbate,targetGene,perturbationFactor,referenceGroup,useGPU)
%D_VIRTUAL_GENE_PERTURBATION
%   Carry out virtual gene perturbation according to the given references.
%   author: Zhaozheng Hou (George)
%
% [perturbatedSample,DE_FoldChange,Pvalue] = D_virtual_gene_perturbation...
%       (Dnet,geneList,samples2perturbate,targetGene,perturbationFactor,...
%           referenceGroup,useGPU)
% parameter:
%   - Dnet: discriminator network to use (network or the name of trained
%       network, such as "BDPP10" for "the discriminator for bulk-RNA seq
%       data with predefined pathways and using 10 references")
%   - geneList: list of genes (strings or numbers), leave blank and skip
%       the matching if the data already matched the gene list
%   - samples2perturbate: expression of the samples for preturbation, each
%       column is one sample.
%   - targetGene: the gene to preturbate.
%   - perturbationFactor: (optional) the factor of perturbation, less than
%       1 for knock out (KO) and larger than 1 for over expression (OE),
%       0.2 by default
%   - referenceGroup: (optional) expression of referencing samples, each
%       column is one sample.
%   - useGPU: (optional) whether using GPU for the processing or not, true
%       by default.
% output:
%   - perturbatedSample: data with the same gene list as reference, each column is one
%       pseudo sample. (unmatched genes get NA)
%   - pseudoNet: data with the gene list same as the network.

%% validate
% Dnet
if (isa(Dnet,"dlnetwork"))
    validateattributes(Dnet,{'dlnetwork'},{'scalar'});
else
    validateattributes(Dnet,{'string','char'},{'scalartext'});
    Dnet=load("core\"+string(Dnet)+".mat",string(Dnet));
    Dnet=struct2cell(Dnet);
    Dnet=Dnet{1};
end
n=Dnet.getLayer(Dnet.InputNames{1}).InputSize;
% geneList
if (isempty(geneList))
    geneList=nan(n(1),1);
    netGeneID=1:n(1);
    listGeneID=netGeneID;
else
    validateattributes(geneList,{'numeric','char','string'},{'vector'});
    [netGeneID,listGeneID] = match_gene_list(geneList);
end
% samples2perturbate
if (~isempty(samples2perturbate))
    validateattributes(samples2perturbate,{'numeric'},{'nonnegative','size',[numel(geneList),nan]});
    temp=zeros(n(1),size(samples2perturbate,2),'single');
    temp(netGeneID,:)=samples2perturbate(listGeneID,:);
    samples2perturbate=temp;
else
    samples2perturbate=zeros(n(1),0,'single');
end
% targetGene
[temp,~,targetGene] = match_gene_list(targetGene);
validateattributes(targetGene,{'logical'},{'scalar','positive'});
targetGene=temp;
invalidSample=(samples2perturbate(targetGene,:)==0);
if any(invalidSample)
    if all(invalidSample)
        error("Target gene is not expressed in any perturbated samples.");
    else
        warning("Target gene is not expressed in some perturbated samples.");
    end
end
% perturbationFactor
if (nargin>=5)
    validateattributes(perturbationFactor,{'numeric'},{'nonnegative','scalar'});
else
    perturbationFactor=0.2;
end
% referenceGroup
if (nargin<6)
    referenceGroup=samples2perturbate;
elseif (isempty(referenceGroup))
    referenceGroup=samples2perturbate;
end
validateattributes(referenceGroup,{'numeric'},{'nonnegative','size',[numel(geneList),nan]});
sizeOfReference=size(referenceGroup,2);
validateattributes(sizeOfReference,{'numeric'},{'>=',2});
temp=zeros(n(1),sizeOfReference,'single');
temp(netGeneID,:)=referenceGroup(listGeneID,:);
referenceGroup=temp;

if (~isempty(samples2perturbate))
    valid=any(samples2perturbate>0,2)&any(referenceGroup>0,2);
    samples2perturbate(~valid,:)=0;
    referenceGroup(~valid,:)=0;
end

% useGPU
if (nargin>=7)
    validateattributes(useGPU,{'numeric','logical'},{'binary'});
    if (isempty(useGPU))
        useGPU=true;
    end
else
    useGPU=true;
end

if (~Dnet.Initialized)
    warning("The network is not yet initialized. Initializing now...");
    Dnet=Dnet.initialize;
end

%% process

% check simiarity
Zref=dlarray(nan(n(1),21,1,10,'single'),'SSCB');

validReference=find(referenceGroup(targetGene,:)>0);
temp=numel(validReference);
if (temp<3)
    error("No enough referncing samples (>=3) express the perturbated gene");
end

for ind=1:10
    Zref(:,:,1,ind)=referenceGroup(:,randi(sizeOfReference,[1,21]));
end
if (useGPU)
    Z=gpuArray(Zref);
else
    Z=Zref;
end

keepID=ones(11,1,'single');
for ind=1:10
    while true
        Z(:,1,1,:)=repmat(referenceGroup(:,validReference(keepID(ind))),...
            [1,1,1,10]);
        temp2=predict(Dnet,Z,Outputs='norm_4');
        Z(targetGene,1,1,:)=perturbationFactor*Z(targetGene,1,1,:);
        temp2=temp2-predict(Dnet,Z,Outputs='norm_4');

        keepID(ind+1)=keepID(ind)+1;
        if (mean(temp2)>0)
            keepID(ind)=keepID(ind+1);
        else
            break;
        end

        if (keepID(ind)>temp)
            break;
        end
    end
    if (keepID(ind)>temp)
        if (ind==1)
            error("This combination of referenceGroup and discriminator is not sensitive to the preturbation.");
        else
            for tid=ind:10
                keepID(tid)=keepID(mod(tid,ind-1)+1);
            end
            break;
        end
    end
end
for ind=1:10
    Zref(:,1,1,ind)=referenceGroup(:,validReference(keepID(ind)));
end



baseRef=predict(Dnet,Z,Outputs='norm_4');
beforePerturbation=nan(10,size(samples2perturbate,2),'single');
afterPerturbation=nan(10,size(samples2perturbate,2),'single');
for ind=1:size(samples2perturbate,2)
    Z(:,1,1,:)=repmat(samples2perturbate(:,ind),[1,1,1,10]);
    beforePerturbation(:,ind)=predict(Dnet,Z,Outputs='norm_4');
    Z(targetGene,1,1,:)=perturbationFactor*Z(targetGene,1,1,:);
    afterPerturbation(:,ind)=predict(Dnet,Z,Outputs='norm_4');
end

if (max(beforePerturbation(:,:,1),[],'all')<min(baseRef))
    warning("Perturbation samples seem significantly different from the referenceGroup...");
end
if (median(beforePerturbation(:,~invalidSample)-afterPerturbation(:,~invalidSample),"all")<=0)
    warning("This discriminator is not sensitive to the preturbation, better to change it...");
end

% knockout differenation
if (~samples2perturbate)
    valid=any(samples2perturbate>quantile(referenceGroup,0.25,'all'),2);
else
    valid=any(referenceGroup>quantile(referenceGroup,0.25,'all'),2);
end
valid(targetGene)=false;

perturbatedSample=samples2perturbate;
perturbatedSample(targetGene,:)=perturbationFactor*perturbatedSample(targetGene,:);

terb=permute(16.^(-1:0.1:1),[1,3,4,2]);
terb012=[ones(numel(terb),1),terb(:),terb(:).^2];
tempMatrix=kron(eye(numel(terb)),0.1*ones(10,1));
terbExpand= cell2mat(arrayfun(@(id) {repmat(terb(id),[1,1,1,10])},permute(1:numel(terb),[1,3,4,2])));
Z=repmat(Z,[1,1,1,numel(terb)]);
DE_log2FC=nan(n(1),1,'single');
Pvalue=nan(n(1),1,'single');

for ind=1:n(1)
    disp(num2str([ind,n(1)],"Processing %u out of %u features..."));

    tempDE=nan(size(samples2perturbate,2)+10,1,'single');
    if (valid(ind))
        for indS=1:size(samples2perturbate,2)
            Z(:,1,1,:)=repmat(samples2perturbate(:,indS),[1,1,1,10*numel(terb)]);
            if (Z(targetGene,1,1,1)>0)
                if (Z(ind,1,1,1)>0)
                    beforePerturbation=predict(Dnet,Z(:,:,1,1:10),Outputs='norm_4'); % TODO: replace this outputs='test'

                    Z(targetGene,1,1,:)=perturbationFactor*Z(targetGene,1,1,:);
                    Z(ind,1,1,:)=terbExpand.*Z(ind,1,1,:);
                    afterPerturbation=predict(Dnet,Z,Outputs='norm_4');
                    ddiff=tempMatrix'*...
                        (afterPerturbation(:)-repmat(beforePerturbation(:),[numel(terb),1])).^2;

                    if (mean(afterPerturbation(tempMatrix(:,end/2+0.5)>0))>mean(beforePerturbation))
                        va=1;
                    else
                        [temp,minid]=min(ddiff);
                        if (temp~=ddiff(end/2+0.5))
                            minid=min(max(minid,3),numel(terb)-2);
                            factors=terb012(minid+(-2:2),:)\ddiff(minid+(-2:2)).extractdata;
                            if (factors(3)>0)
                                va=clip(-factors(2)/(2*factors(3)),terb(minid-2),terb(minid+2));
                            else
                                va=terb(minid);
                            end
                        else
                            va=1;
                        end
                    end
                else
                    va=1;
                end
                tempDE(indS)=va;
                perturbatedSample(ind,indS)=tempDE(indS)*perturbatedSample(ind,indS);
            end
        end

        for indS=1:10
            Z(:,1,1,:)=repmat(Zref(:,1,1,indS),[1,1,1,10*numel(terb)]);
            if (Z(targetGene,1,1,1)>0)
                if (Z(ind,1,1,1)>0)
                    beforePerturbation=predict(Dnet,Z(:,:,1,1:10),Outputs='norm_4'); % TODO: replace this outputs='test'

                    Z(targetGene,1,1,:)=perturbationFactor*Z(targetGene,1,1,:);
                    Z(ind,1,1,:)=terbExpand.*Z(ind,1,1,:);
                    afterPerturbation=predict(Dnet,Z,Outputs='norm_4');
                    ddiff=tempMatrix'*...
                        (afterPerturbation(:)-repmat(beforePerturbation(:),[numel(terb),1])).^2;

                    if (mean(afterPerturbation(tempMatrix(:,end/2+0.5)>0))>mean(beforePerturbation))
                        va=1;
                    else
                        [temp,minid]=min(ddiff);
                        if (temp~=ddiff(end/2+0.5))
                            minid=min(max(minid,3),numel(terb)-2);
                            factors=terb012(minid+(-2:2),:)\ddiff(minid+(-2:2)).extractdata;
                            if (factors(3)>0)
                                va=min(max(-factors(2)/(2*factors(3)),terb(minid-2)),terb(minid+2));
                            else
                                va=terb(minid);
                            end
                        else
                            va=1;
                        end
                    end
                else
                    va=1;
                end
                tempDE(size(samples2perturbate,2)+indS)=va;
                %perturbatedSample(ind,indS)=tempDE(indS)*perturbatedSample(ind,indS);
            end
        end

        tempDE(isnan(tempDE))=[];
        tempDE=log2(tempDE);
        DE_log2FC(ind)=mean(tempDE);
        [~,Pvalue(ind)]=ttest(tempDE);
    end
end

%% formating
temp=nan(numel(geneList),size(perturbatedSample,2));
temp(listGeneID,:)=perturbatedSample(netGeneID,:);
perturbatedSample=temp;

temp=nan(numel(geneList),1,'single');
temp(listGeneID)=DE_log2FC(netGeneID);
DE_log2FC=temp;
temp=nan(numel(geneList),1,'single');
temp(listGeneID)=Pvalue(netGeneID);
Pvalue=temp;
end