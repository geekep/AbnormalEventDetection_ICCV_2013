params.H = 120;       % loaded video height size
params.W = 160;       % loaded video width size

params.patchWin = 10; % 3D patch spatial size
params.tprLen = 5;    % 3D patch temporal length

params.BKH = params.H / params.patchWin;      % region number in height
params.BKW = params.W / params.patchWin;      % region number in width

params.srs = 5;       % spatial sampling rate in training video volume
params.trs = 2;       % temporal sampling rate in training video volume

params.PCAdim = 100;  % PCA Compression dimension

params.MT_thr = 5;    % 3D patch selecting threshold