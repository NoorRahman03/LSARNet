  % Load test images from the SOC dataset folder
  % Labels are automatically assigned based on subfolder names (one folder per class)
  dataset_path ='C:\Users\mahsu\OneDrive\Desktop\SARATR\Dataset\SOC1\'
  data=fullfile(dataset_path,'Test');
  testDataSOC3=imageDatastore(data,'IncludeSubfolders',true,'LabelSource','foldernames');

  % Load the pre-trained LSARNet model from the saved .mat file
  load LSARNet.mat

  % Run the trained model on each test image and collect predicted class labels
  allclass=[];
  for i = 1:length(testDataSOC3.Labels)
    a=readimage(testDataSOC3,i);
    out=classify(training,a);
    allclass=[allclass out];
  end

  % Store predictions and plot confusion matrix to evaluate classification performance
  Predicted=allclass;
  figure,
  plotconfusion(testDataSOC3.Labels,Predicted')
