%% demo for figure 6
%% preprocessing
clear;clc;
DRM_path = '../Psychol_Rec/DRM.xlsx';

DRM = readtable(DRM_path,'PreserveVariableNames',true);
subnum = length(unique(DRM{:,2}));

data = cell(1,subnum);
id_add = 1;
for id = 1:max(DRM{:,2})
    index = find(DRM{:,2}==id);
    if size(index,1) > 0
        data{id_add} = DRM{index(1:size(index,1)),6:15};
        id_add = id_add + 1;
    end
end
save('DRM_emo','data','DRM');

%% correlation
load('DRM_emo');
origin_label = {'upset','hostile','alert','ashamed','inspired','nervous','determined','attentive','afraid','active'};
emotion_label = {'upset','hostile','ashamed','nervous','afraid','alert','inspired','determined','attentive','active'};
trans_label = [1,2,4,6,9,3,5,7,8,10]; % let PA in first 5, NA in last 5
R_sub = zeros(143,10,10);
P_sub = zeros(143,10,10);
count = 0;
for subno = 1:size(data,2)
    temp = data{subno};
    [r,p] = corrcoef(temp(:,trans_label));
    r(logical(eye(size(r)))) = 0;
    if ~isempty(find(r==1))
        continue;
    end
    if isequal(ismissing(r),zeros(10,10))
        count = count + 1;
        R_sub(count,:,:) = r;
        P_sub(count,:,:) = p;
    end
end
R = squeeze(sum(R_sub,1)/count);
P = squeeze(sum(P_sub,1)/count);
figure;
R(logical(eye(size(R)))) = 0;
subplot(121);
heatmap(emotion_label,emotion_label,R);
% imagesc(R);
% set(gca,'XTickLabel',emotion_label);
% set(gca,'YTickLabel',emotion_label,'FontSize',6);
title('correlation r');
colorbar;
subplot(122);
z_r = 0.5*log((1+R_sub)./(1-R_sub));
for i = 1:10
    for j = 1:10
        [~,p_z(i,j)] = ttest(z_r(1:count,i,j),0,'Alpha',0.05);
    end
end
p_z(logical(eye(size(p_z)))) = 1;
heatmap(emotion_label,emotion_label,-log10(p_z));
title('-log p');
colorbar;


