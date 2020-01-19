%% Parameters
set_params;

H = params.H;
W = params.W;
patchWin = params.patchWin;
tprLen = params.tprLen;
BKH = params.BKH;
BKW = params.BKW;
PCAdim = params.PCAdim;
trainFileNum = 16;

addpath('functions')
addpath('data')

%% Training feature generation (about 1 minute)

tic;
% The maximum sample number in each training video is 10000
numEachVol = 10000;
Cmatrix = zeros(tprLen * patchWin ^ 2, 6 * numEachVol);
rand('state', 0);

% training using normal videos
for ii = [1 : 10, 12, 13]
    [feaRawTrain, LocV3Train] = train_features(['data/CV_Normal_', num2str(ii), '.mat'], params);
    t = randperm(size(feaRawTrain, 2));
    curFeaNum = min(size(feaRawTrain, 2), numEachVol);
    % put random curFeaNum column into Cmatrix
    Cmatrix(:, numEachVol * (ii - 1) + 1 : numEachVol * (ii - 1) + curFeaNum) = feaRawTrain(:, t(1 : curFeaNum));
    disp(['Feature extraction in video ', num2str(ii), ' is done!'])
end

% take out the zero valued columns
Cmatrix(:, sum(abs(Cmatrix)) == 0) = [];

% compress raws
COEFF = PCA(Cmatrix');
Tw = COEFF(:, 1 : PCAdim)';
feaMatPCA = Tw * Cmatrix;
save('data/sparse_combinations/Tw.mat', Tw');
toc;

%% Sparse combination learning  (about 4 minutes)

tic;
D = sparse_combination(feaMatPCA, 20, 0.21);
% input:
%   @X: feature matrix m x N (m is feature dimension, N is feature number)
%   @Dim: dimension of a combination
%   @Thr: lambda in paper
%
% output:
%   @D: sparse combination

for ii = 1 : length(D);
    % R matrix in Eq. (13).
    R(ii).val = D(ii).val * inv(D(ii). val' * D(ii). val) * D(ii).val' - eye(size(D(ii).val, 1));
end

save('data/sparse_combinations/D.mat', 'D');
save('data/sparse_combinations/R.mat', 'R');
toc;