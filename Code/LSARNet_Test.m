
dataset_path ='C:\Users\mahsu\OneDrive\Desktop\SARATR\Dataset\SOC1\'
data=fullfile(dataset_path,'Test');
testDataSOC3=imageDatastore(data,'IncludeSubfolders',true,'LabelSource','foldernames');

load LSARNet.mat

% %for single File.
% [fileName,filePath] = uigetfile('*.*', "Select an Image");
% a=imread(fullfile(filePath,fileName));
% %imshow(a)
% [out,score]=classify(training,a);
% figure,
% imshow(a)
% title(string(3out));


allclass=[];
for i = 1:length(testDataSOC3.Labels)
a=readimage(testDataSOC3,i);
%a=uint8(a); %for binary images
out=classify(training,a);
allclass=[allclass out];
%figure,imshow(a);
%title(string(out));
end

Predicted=allclass;
figure,
plotconfusion(testDataSOC3.Labels,Predicted')

% sourcePath=('C:\Users\mahsu\OneDrive\Desktop\SARATR\Dataset\SOC\TestSet\BMP2\SN_9563\');
% Files = dir([sourcePath,'*.tif']);
% allclass=[];
% for i = 1:length(Files)
% a=imread([sourcePath,Files(i).name]);
% out=classify(training,a);
% arr(i,:)=out;
% fname(i,:)=Files(i).name;
% %figure,imshow(a);
% %title(string(out));
% end



