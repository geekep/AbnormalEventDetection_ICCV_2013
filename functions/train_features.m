function  [feaRawTrain, LocV3Train]  = train_features(fileName, params)
% extract 3D gradient feature for all training video volumes
% input:
%   fileName: filename of a file containing all training video volumes
%   params: parameters
%
% output:
%   feaRawTrain: a m * N matrix whose column is m dimemsions 3D gradient feature. (N in total)
%   LocV3Train: a 3 * M matrix records locations of all 3D gradient features. (N in total)

H = params.H;
W = params.W;
patchWin = params.patchWin;
srs = params.srs;
trs = params.trs;
MT_thr = params.MT_thr;
tprLen = params.tprLen;

load(fileName, 'Video_Output');

for ii = 1 : size(Video_Output, 4)
    % Video_Output: rgb image
    Video_Output(:, :, :, ii) = Video_Output(:, :, :, ii) / 255;
    % vol: gray image
    vol(:, :, ii) = rgb2gray(Video_Output(:, :, :, ii));
end

voBlur = vol;
blurKer = fspecial('gaussian', [3,3], 1);
mask = conv2(ones(H,W), blurKer, 'same'); % eliminate the dark border

for pp = 1 : size(vol, 3)
    voBlur(:, :, pp) = conv2(vol(:, :, pp), blurKer, 'same') ./ mask;
end

% calculate gradient of time
volG = abs(voBlur(:, :, 1 : (end-1)) - voBlur(:, :, 2 : end));

rsNum = 10000; % reserved number:

count = 0;
motionReg = zeros(size(volG));
motionResponse = zeros(size(volG));

for frameID = (tprLen + 1) : (size(volG, 3) - tprLen)
    motionReg(:, :, frameID) = conv2(volG(:, :, frameID), ones(patchWin), 'same');
end

for frameID = (tprLen + 1) : (size(volG, 3) - tprLen)
    % accumulate the adjacent 5 frames
    motionResponse(:, :, frameID) = sum(motionReg(:, :, frameID - 2 : frameID + 2), 3);
end

feaRawTrain = zeros(tprLen * patchWin ^ 2, rsNum);
LocV3Train = zeros(3, rsNum);

% samlping in temporal axis
for frameID = (tprLen + 1) : trs : (size(volG, 3) - tprLen)
    % samlping in spatial axis
    for ii = round(patchWin / 2) + 1 : srs : H - round(patchWin / 2)
        for jj = round(patchWin / 2) + 1 : srs : W - round(patchWin / 2)
            tmp = motionResponse(ii, jj, frameID);
            % Those volumes that contains little motion information are
            % abandoned.
            if tmp > MT_thr
                count = count + 1;
                cube = volG(ii - floor(patchWin / 2) : ii + floor(patchWin / 2) - 1, jj - floor(patchWin / 2) : jj + floor(patchWin / 2) - 1, frameID - 2 : frameID + 2);
                % expand by column
                feaRawTrain(:, count) = cube(:);
                LocV3Train(:, count) =  [ii; jj; frameID]';
            end
        end
    end
end

delIdx = find(sum(LocV3Train) == 0);
% take out all-zero columns
feaRawTrain(:, delIdx) = [];
LocV3Train(:, delIdx) = [];

feaRawTrain = bsxfun(@rdivide, feaRawTrain, sqrt(sum(feaRawTrain .^ 2)));

end