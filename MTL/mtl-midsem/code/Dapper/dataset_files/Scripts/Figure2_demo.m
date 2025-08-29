
%% demo for figure 2
clear;clc;

ESM_path = '../Psychol_Rec/ESM.xlsx';
ESM = readtable(ESM_path,'PreserveVariableNames',true);
time = ESM{:,2};
label = datetime('00:00','Format','HH:mm')+hours(0:0.5:23.5);
count = zeros(1,48);
subplot(211);
for i = 1:size(time,1)
    temp = time{i,1};
    time2 = datetime(temp(12:16),'Format','HH:mm');
    time1 = time2-hours(0.5);
    label1 = label <= time1-hours(0.5);
    label2 = label >= time2;
    count = count + abs(label1 + label2 - 1);
end
%bar(0:0.5:23.5,count,'FaceColor',[0.4,0.5,0.7]);
bar(6:0.5:23.5,count(13:end),'FaceColor',[0.4,0.5,0.7]);
xlabel('Time (o''clock)');
ylabel('Event count');
title('Distribution of ESM events');

DRM_path = '../Psychol_Rec/DRM.xlsx';
DRM = readtable(DRM_path,'PreserveVariableNames',true);
time = DRM{:,5};
label = datetime('00:00','Format','HH:mm')+hours(0:0.5:23.5);
count = zeros(1,48);
count2 = 0;
subplot(212);
for i = 1:size(time,1)
    temp = time{i,1};
    if isempty(temp)
        continue;
    end
    time1 = datetime(temp(1:5),'Format','HH:mm');
    time2 = datetime(temp(7:11),'Format','HH:mm');
    label1 = label <= time1-hours(0.5);
    label2 = label >= time2;
    if time1 < datetime('06:00','Format','HH:mm')
        count2 = count2+1;
    end
    count = count + abs(label1 + label2 - 1);
end
%bar(0:0.5:23.5,count,'FaceColor',[0.4,0.5,0.7]);
bar(6:0.5:23.5,count(13:end),'FaceColor',[0.4,0.5,0.7]);
xlabel('Time (o''clock)');
ylabel('Event count');
title('Distribution of DRM events');

