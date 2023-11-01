% load the data from the dataset-letters.mat file
loadData= load('dataset-letters.mat')

imageInfo=loadData.dataset.images;
labelInfo=loadData.dataset.labels;

% convert to double type;

imageInfo=double(imageInfo);

% reshape the image to 28x28 
numOfImages = size(imageInfo,1);
reshapedImages= zeros(numOfImages, 28,28);

for i =1: numOfImages
    reshapedImages(i,:,:)= reshape(imageInfo(i,:), 28,28);
end


% employ a vectorized approach.
randomIndices= randperm(size(reshapedImages,1),12);

% we create a figure
figure;
for i = 1:12
    subplot(3,4,i);
    % display image with labels  
    imshow(squeeze(reshapedImages(randomIndices(i),:,: )));
    title(['Label: ' num2str(labelInfo(randomIndices(i)))]);
    fprintf("");
end

% save to PNG file
ExampleName = 'Example.png';
saveas(gcf,ExampleName, 'png');

% Need to split data into training and testing data.
TotalNumOfImages=size(reshapedImages,1);
randomIndices= randperm(TotalNumOfImages);

% 50% will be assigned to training and the other 50% to testing
halfNumImages= round(TotalNumOfImages/2)

% Training set - a subset
imageTraining= imageInfo(randomIndices(1:halfNumImages), :,:);
labelTraining= labelInfo(randomIndices(1:halfNumImages),:);

% Testing set - a subsets
imageTesting=imageInfo(randomIndices(halfNumImages + 1:end), :,:);
labelTesting= labelInfo(randomIndices(halfNumImages + 1: end), :);

% Checking the label distribution
trainingLabelDis=histcounts(labelTraining);
testingLabelDis=histcounts(labelTesting);

fprintf('Label distribution for the training set - a subset');
disp(trainingLabelDis);
fprintf('Label distribution for the testing test - a subset');
disp(testingLabelDis);
fprintf('\n');
% We can train the KNN model using euclidean distance
%Measure time taken to train model
tic;
knnModelEuclidean= fitcknn(imageTraining, labelTraining);
trainingtimeEuclidean=toc;

% Prediction using the resulting KNN Model for euclidean distance
% Measure time taken to at testing stage for euclidean distance
tic;
predictEuclidean=predict(knnModelEuclidean,imageTesting );
testingtimeEuclidean=toc;

% Accuracy for Euclidean distance.
calculateEuclidean=sum(predictEuclidean==labelTesting)/numel(labelTesting);

% results of euclidean distance
disp("Model is KNN Model using the euclidean distance metric");
disp("Accuracy: "+  calculateEuclidean);
disp("Training time: "+ trainingtimeEuclidean);
disp("Testing time: "+ testingtimeEuclidean);
fprintf('\n');

% We can train the KNN model using cosine 
% Measure time taken to train the model
tic;
knnModelCosine=fitcknn(imageTraining, labelTraining);
trainingTimeCosine=toc;

% Prediction using the resulting KNN Model for the cosine distance
% Measure time taken to test for cosine distance
tic;
predictCosine=predict(knnModelCosine, imageTesting);
testingTimeCosine=toc;

% Accuracy for cosine distance.
calculateCosine=sum(predictCosine==labelTesting)/numel(labelTesting);

disp("Model is KNN Mode using the cosine distance metric");
disp("Accuracy: "+ calculateCosine);
disp("Training time: "+ trainingTimeCosine);
disp("Testing time: "+ testingTimeCosine);
fprintf('\n');

% we use the SVM model as a means for comparison
% Measure the time taken to train the model
tic;
svmModel=fitcecoc(imageTraining, labelTraining);
trainingTimeSVM=toc;

% prediction for the svm model
tic;
predictSvm=predict(svmModel, imageTesting);
testingTimeSVM=toc;

% Accuracy of the SVM Model
calculateSVM=sum(predictSvm==labelTesting)/numel(labelTesting);

disp("Model is SVM and used as a means for comparison with KNN model");
disp("Accuracy: "+ calculateSVM);
disp("Training time: "+ trainingTimeSVM);
disp("Testing time: " + testingTimeSVM);
fprintf('\n');
% we use the ensemble model as a means for comparison
tic;
ensembleModel=fitcensemble(imageTraining, labelTraining);
trainingTimeEnsemble=toc;
% prediction for the ensemble model
tic;
ensemblePredict= predict(ensembleModel, imageTesting);
testingTimeEnsemble=toc;

calculateEnsemble = sum(ensemblePredict==labelTesting)/numel(labelTesting);

disp("Model is Ensemble and used as a means for comparison with KNN model");
disp("Accuracy: "+ calculateEnsemble);
disp("Training time: "+ trainingTimeEnsemble);
disp("Testing time: "+ testingTimeEnsemble);
fprintf('\n');
