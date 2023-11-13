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

% We can train the KNN model using euclidean distance
% initialize an appropriate value of k
% 
k=5
%   initialize array to store predicted labels
 % measure computation time for L2
 tic;
predictEuclidean = zeros(size(labelTesting));



for i = 1:size(imageTesting,1)
%Define our parameters 
    comp1=imageTraining;
    comp2 = repmat(imageTesting(i,:), [size(imageTraining,1),1]);
%calculate euclidean distance
    Euclideandistance = sqrt(sum((comp1-comp2).^2,2));
%sort the distance
    [~,ind]=sort(Euclideandistance);
    indSort=ind(1:k);
    labs=labelTraining(ind);
     predictEuclidean(i) = mode(labs);
end

computationTimeL2=toc;
%calculate euclidean accuracy and confusion chart
calculateEuclideanAccuracy=sum(labelTesting==predictEuclidean)/size(labelTesting,1);
chart1= confusionchart(labelTesting, predictEuclidean);

%results printed to terminal to check
disp("Model is KNN Model using the Euclidean distance metric");
disp("Accuracy: "+ calculateEuclideanAccuracy);
disp("Computation time: "+ computationTimeL2);

%Carry out K-NN model classification with l1 distance
%  measure computation time for L1
tic;
% define our prediction 
predictL1 = zeros(size(labelTesting,1),1);

 for i = 1:size(imageTesting,1)
     comp1=imageTraining;
     comp2=repmat(imageTesting(i,:), [size(imageTraining,1),1]);
%calculate the l1 distance
     distanceL1 = sum(abs(comp1-comp2),2);
     [~,indL1]=sort(distanceL1);
     indL1=indL1(1:k);
     labs=labelTraining(indL1);
     predictL1(i) = mode(labs);
 end
 
 computationTimel1=toc;
%Accuracy is calculated
 calculateL1Accuracy=sum(labelTesting==predictL1)/size(labelTesting,1);
 chart2 = confusionchart(labelTesting, predictL1);
% display all the results
disp("Model is KNN Model using the L1 metric");
disp("Accuracy: "+ calculateL1Accuracy);
disp("Computation time: "+ computationTimel1);

%save all results in .mat file for analysis later
save('KNN_results.mat', 'predictEuclidean', 'computationTimeL2','calculateEuclideanAccuracy','predictL1', 'computationTimel1', 'calculateL1Accuracy');

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
calculateSVMAccuracy=sum(labelTesting==predictSvm)/size(labelTesting,1);
chart3 = confusionchart(labelTesting, predictSvm);

disp("Model is SVM and used as a means for comparison with KNN model");
disp("Accuracy: "+ calculateSVMAccuracy);
disp("Training time: "+ trainingTimeSVM);
disp("Testing time: " + testingTimeSVM);


% we use the ensemble model as a means for comparison
tic;
ensembleModel=fitcensemble(imageTraining, labelTraining);
trainingTimeEnsemble=toc;
% prediction for the ensemble model
tic;
ensemblePredict= predict(ensembleModel, imageTesting);
testingTimeEnsemble=toc;

%calculate ensemble accuracy
calculateEnsembleAccuracy = sum(labelTesting==ensemblePredict)/size(labelTesting,1);
chart3 = confusionchart(labelTesting, ensemblePredict);
%display results in terminal as part of testing 
disp("Model is Ensemble and used as a means for comparison with KNN model");
disp("Accuracy: "+ calculateEnsembleAccuracy);
disp("Training time: "+ trainingTimeEnsemble);
disp("Testing time: "+ testingTimeEnsemble);

%save all results for analysis later
save('ModelResults.mat','predictSvm' ,'trainingTimeSVM','testingTimeSVM', 'calculateSVMAccuracy','ensemblePredict', 'trainingTimeEnsemble', 'testingTimeEnsemble', 'calculateEnsembleAccuracy');

