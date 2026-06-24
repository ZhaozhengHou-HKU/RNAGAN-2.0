function netG = GeneratorNetwork_V2(pathwayMatrix,numRef)
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

netG = dlnetwork;

tempNet=layerNormalizationLayer("Name","norm_GeneLevel");
tempNet.Offset=repmat(log(exp(1)-1),[numGene,1]);
tempNet.OffsetLearnRateFactor=0.01;
tempNet = [
    imageInputLayer([numGene,numRef,1],"Name","Input","Normalization","none");
    reshapeLayer("formattingLayer_1",[3 2 1 4],"SSCB");
    depthConcatenationLayer(2,"Name","depthcat");
    globalAveragePooling2dLayer("Name","gapool");
    concatenationLayer(2,2,"Name","concat_1");
    %%layerNormalizationLayer("Name","norm_0");
    reshapeLayer("formattingLayer_2",[3 2 1 4],"SSCB");
    fullyConnectedLayer(16*featureSize,"Name","fc_1");
    layerNormalizationLayer("Name","norm_1");
    dropoutLayer(0.5,"Name","dropout");
    swishLayer("Name","swish_1");
    fullyConnectedLayer(4*featureSize,"Name","fc_2");
    layerNormalizationLayer("Name","norm_2");
    swishLayer("Name","swish_2");
    fullyConnectedLayer(featureSize,"Name","fc_3");
    layerNormalizationLayer("Name","latentSpace");
    quadraticLayer("Name","quadratic");
    fullyConnectedLayer(4*featureSize,"Name","fc_4");
    layerNormalizationLayer("Name","norm_4");
    swishLayer("Name","swish_4");
    concatenationLayer(1,2,"Name","concat_2");
    fullyConnectedLayer(16*featureSize,"Name","fc_5");
    layerNormalizationLayer("Name","norm_5");
    swishLayer("Name","swish_5");
    concatenationLayer(1,2,"Name","concat_3");
    fullyConnectedLayer(numGene,"Name","fc_GeneLevel");
    tempNet;
    softplusLayer("Name","softplus");
    reshapeLayer("formattingLayer_3",[3 4 1 2],"SSCB");
    multiplicationLayer(2,"Name","multiplication");
    reshapeLayer("formattingLayer_4",[3 2 1 4],"SSCB");
    concatenationLayer(2,2,"Name","concat")];
if (numPathways==0)
    tempNet(3)=[];%depthcat
end
netG = addLayers(netG,tempNet);
netG = connectLayers(netG,"swish_1","concat_3/in2");
netG = connectLayers(netG,"swish_2","concat_2/in2");

netG = addLayers(netG,globalMaxPooling2dLayer("Name","gmpool"));
if (numPathways==0)
    netG = connectLayers(netG,"formattingLayer_1","gmpool");
else
    netG = connectLayers(netG,"depthcat","gmpool");
end
netG = connectLayers(netG,"gmpool","concat_1/in2");
netG = addLayers(netG,globalAveragePooling2dLayer("Name","gapool_2"));
netG = connectLayers(netG,"formattingLayer_1","gapool_2");
netG = connectLayers(netG,"gapool_2","multiplication/in2");
netG = connectLayers(netG,"Input","concat/in2");

if (numPathways>0)
    tempNet = convolution2dLayer([numGene,1],numPathways, ...
        'Name','Pathways',"BiasLearnRateFactor",0);
    % load pathway info
    if (any(pathwayMatrix,'all'))
        tempNet.WeightLearnRateFactor=0;
        tempNet.Weights=single(permute(full(pathwayMatrix),[2,3,4,1]));
    end
    netG = addLayers(netG,tempNet);
    netG = connectLayers(netG,"Input","Pathways");
    netG = connectLayers(netG,"Pathways","depthcat/in2");
end

netG = initialize(netG);
end