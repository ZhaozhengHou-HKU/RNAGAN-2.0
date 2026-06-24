function netD = DiscriminatorNetwork_V2(pathwayMatrix,numRef)
%provide pathway in [pathway,genes]
%learnable pathways with all true
%no pathway with an empty matrix with 0 rows, or as the number of genes

if (isscalar(pathwayMatrix))
    numPathways=0;
    numGene=pathwayMatrix;
else
    [numPathways,numGene]=size(pathwayMatrix);
end

featureSize=16;

netD = dlnetwork;

tempNet = [
    imageInputLayer([numGene,numRef+1,1],"Name","Input","Normalization","none");
    reshapeLayer("formattingLayer_1",[3 2 1 4],"SSCB");%spaceToDepthLayer([numGene,1],"Name","formattingLayer_1");%
    depthConcatenationLayer(2,"Name","depthcat");
    dataSplitLayer(1,"split_first");
    concatenationLayer(2,3,"Name","concat_1");
    instanceNormalizationLayer("Name","norm_0");%instanceNormalizationLayer("Name","instancenorm_0");
    reshapeLayer("formattingLayer_2",[3 2 1 4],"SSCB");%depthToSpace2dLayer([numGene+numPathways,1],"Name","formattingLayer_2","Mode","crd");%
    fullyConnectedLayer(16*featureSize,"Name","fc_1");
    layerNormalizationLayer("Name","norm_1");
    dropoutLayer(0.5,"Name","dropout");
    swishLayer("Name","swish_1");
    fullyConnectedLayer(4*featureSize,"Name","fc_2");
    layerNormalizationLayer("Name","norm_2");
    swishLayer("Name","swish_2");
    fullyConnectedLayer(featureSize,"Name","fc_3");
    quadraticLayer("Name","quadratic");
    fullyConnectedLayer(1,"Name","fc_4");
    batchNormalizationLayer("Name","norm_4");
    sigmoidLayer("Name","sigmoid")];
if (numPathways==0)
    tempNet(3)=[];%depthcat
end
netD = addLayers(netD,tempNet);

tempNet = [
    dataSplitLayer(2:(numRef+1),"split_reference");
    globalAveragePooling2dLayer("Name","gapool")];
netD = addLayers(netD,tempNet);
netD = addLayers(netD,globalMaxPooling2dLayer("Name","gmpool"));
if (numPathways==0)
    netD = connectLayers(netD,"formattingLayer_1","split_reference");
else
    netD = connectLayers(netD,"depthcat","split_reference");
end
netD = connectLayers(netD,"split_reference","gmpool");
netD = connectLayers(netD,"gmpool","concat_1/in3");
netD = connectLayers(netD,"gapool","concat_1/in2");

if (numPathways>0)
    tempNet = convolution2dLayer([numGene,1],numPathways, ...
        'Name','Pathways',"BiasLearnRateFactor",0);
    % load pathway info
    if (any(pathwayMatrix,'all'))
        tempNet.WeightLearnRateFactor=0;
        tempNet.Weights=single(permute(full(pathwayMatrix),[2,3,4,1]));
    end
    netD = addLayers(netD,tempNet);
    netD = connectLayers(netD,"Input","Pathways");
    netD = connectLayers(netD,"Pathways","depthcat/in2");
end

rng(0);
netD = initialize(netD);
end