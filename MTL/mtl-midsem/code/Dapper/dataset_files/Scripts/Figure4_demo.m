
%% demo for figure 4
%% preprocessing

clear;clc;
ESM_path = '../Psychol_Rec/ESM.xlsx';
DRM_path = '../Psychol_Rec/DRM.xlsx';

ESM = readtable(ESM_path,'PreserveVariableNames',true);
subnum = length(unique(ESM{:,1}));

data = cell(1,subnum);
id_add = 1;
for id = 1:max(ESM{:,1})
    index = find(ESM{:,1}==id);
    if size(index,1) > 0
        data{id_add} = ESM{index(1:size(index,1)),15:24};
        id_add = id_add + 1;
    end
end
save('ESM_emo','data','ESM');

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
%% plot distribution bars

load('ESM_emo');
emotion_label = {'upset','hostile','alert','ashamed','inspired','nervous','determined','attentive','afraid','active'};
rate_mean = zeros(144,10);
count = zeros(10,8);
edges = zeros(10,9);
for i = 1:length(data)
    rate_mean(i,:) = mean(data{i},1);
end
for i = 1:10
    [count(i,:),edges(i,:)] = histcounts(rate_mean(:,i),1:0.5:5);
end
bar3(count,0.7);
xlabel('Self-report scores');
ylabel('Emotion catogories');
title('ESM score distribution');
set(gca,'ydir','normal');
set(gca,'XTickLabel',{'1','1.5','2','2.5','3','3.5','4','4.5','5'});
set(gca,'YTickLabel',emotion_label);

figure;
load('DRM_emo');
emotion_label = {'upset','hostile','alert','ashamed','inspired','nervous','determined','attentive','afraid','active'};
rate_mean = zeros(143,10);
count = zeros(10,8);
edges = zeros(10,9);
for i = 1:length(data)
    rate_mean(i,:) = mean(data{i},1);
end
for i = 1:10
    [count(i,:),edges(i,:)] = histcounts(rate_mean(:,i),1:0.5:5);
end
bar3(count,0.7);
xlabel('Self-report scores');
ylabel('Emotion catogories');
title('DRM score distribution');
set(gca,'ydir','normal');
set(gca,'XTickLabel',{'1','1.5','2','2.5','3','3.5','4','4.5','5'});
set(gca,'YTickLabel',emotion_label);
