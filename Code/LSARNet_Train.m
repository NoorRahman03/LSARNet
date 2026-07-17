% =========================================================
% LSARNet Training Script
% =========================================================

clc
clear all
close all

% Load training images
dataset_path = 'C:\Users\mahsu\OneDrive\Desktop\SARATR\Dataset\SOC1'
data=fullfile(dataset_path,'Train');
trainData=imageDatastore(data,'IncludeSubfolders',true,'LabelSource','foldernames');
count=trainData.countEachLabel;

% Define augmentation strategy — random rotation to improve generalisation
augmenter = imageDataAugmenter( ...
   'RandRotation', [-10, 10] ...  
);
 
% Split dataset into 80% training and 20% validation
[trainingdata validdata]= splitEachLabel(trainData,0.8,'randomized');

% Apply augmentation and resize all images to 128x128 greyscale
augmentedImds = augmentedImageDatastore([128 128 1], trainingdata, 'DataAugmentation', augmenter);

% =========================================================
% Define LSARNet Architecture
% Two convolutional blocks followed by a fully connected classifier
% =========================================================
layers = [
   imageInputLayer([128 128 1]) 
        
   % First convolutional block
   convolution2dLayer(6,8,'Padding','same')     
   batchNormalizationLayer 
   reluLayer
   averagePooling2dLayer(2,'Stride',1) 

   % Second convolutional block
   convolution2dLayer(6,16,'Padding','same')                    
   batchNormalizationLayer 
   reluLayer
   averagePooling2dLayer(2,'Stride',1) 

   % Classifier head — 3 output classes (BMP2, BTR70, T72)
   fullyConnectedLayer(3)
   softmaxLayer
   classificationLayer()
 ]
% analyzeNetwork(net)
% Training
% opt=trainingOptions('sgdm','Maxepoch',10,'InitialLearnRate',0.0001,'Plots','training-progress');

opt = trainingOptions('sgdm', ... % Stochastic Gradient Descent with Momentum
    'MaxEpochs',10, ... % Maximum number of training epochs   
    'Momentum',0.5,...% Mo
    'InitialLearnRate', 0.001, ... % Initial learning rate
    'ValidationData', validdata, ... % validataion 
    'ValidationFrequency', 5,... %validatain data
    'LearnRateSchedule', 'piecewise',...%Learn Rate
    'LearnRateDropFactor', 0.1, ...%learnRate
    'LearnRateDropPeriod', 8, ...%LearnRate Drop period
    'Plots', 'training-progress', ... % Plot training progress during training
    'MiniBatchSize', 16, ... % Size of mini-batches for each iteration
    'Verbose', true); % Display information during training


[training,info]=trainNetwork(augmentedImds,layers,opt)

save('LSARNet.mat', 'training');

%h=findall(groot,'Type','Figure');
%h.MenuBar='figure';
%h.MenuBar='none';
%close(h); %close h after saving the fig

  imgpath="C:\Users\mahsu\OneDrive\Desktop\SARATR\Dataset\SOC\Test\T72\HB03333.tif";
  img=imread(imgpath);
  act = activations(training, img, 'conv_1');% to get features for layers
%  montage(act);

 % =============HEat Map Code-------------------------
 imgpath="C:\Users\mahsu\OneDrive\Desktop\SARATR\Dataset\SOC\Test\T72\HB03333.tif";
  img=imread(imgpath);
  act = activations(training, img, 'conv_2');% to get features for layers
 numFeatureMaps = size(act, 3);
 figure;
for i = 1:numFeatureMaps
    % Extract the i-th feature map
    feature_map = act(:, :, i, 1);
    
    % Create a subplot for the current feature map
    subplot(ceil(numFeatureMaps / 4), 4, i); % Adjust layout as needed
    imagesc(feature_map); % Display the feature map
    colorbar; % Add a color bar
    colormap('jet'); % Use the 'jet' colormap for better visualization
    title(['Feature Map ' num2str(i)]);
    xlabel('X-axis');
    ylabel('Y-axis');
end

% Add a title for the entire figure
sgtitle('HeatMap Layer conv_2');
 

% =============Visualizing Convolutional Kernels-------------------------

% Display kernels of a convolutional layer
kernels = training.Layers(6).Weights; % Assuming 2nd layer is convolutional - conv1=2 and conv2=6
numKernels = size(kernels, 4);
figure;
for i = 1:numKernels
    subplot(ceil(numKernels / 8), 8, i);
    imagesc(kernels(:, :, 1, i)); % Display the i-th kernel (assuming single-channel input)
    colormap('gray');
    colorbar;
    title(['Kernel ' num2str(i)]);
end

%============================================
% Load a sample image
img = imread("C:\Users\mahsu\OneDrive\Desktop\SARATR\Dataset\SOC\Test\T72\HB03333.tif");
img = imresize(img, [128 128]); % Resize to match input size if needed

% Extract the kernels
weights_conv2 = training.Layers(6).Weights;

% Visualize the activation maps for a sample image
numKernels = size(weights_conv2, 4);
figure;
for i = 1:numKernels
    kernel = weights_conv2(:, :, 1, i);
    activation_map = conv2(double(img), double(kernel), 'valid');
    subplot(ceil(numKernels / 4), 4, i);
    imagesc(activation_map);
    colormap('jet');
    colorbar;
    title(['Kernel ' num2str(i)]);
end


 
%save training;
