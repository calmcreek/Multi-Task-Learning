
%% demo for figure 3
clear;clc;

ESM_path = '../Psychol_Rec/ESM.xlsx';
ESM = readtable(ESM_path,'PreserveVariableNames',true);
label1 = {'classroom','library','dormitory','playground','gym','canteen','department','on-campus','home','internship','off-campus'};
label2 = {'self','teacher','classmate','families','stranger'};
label3 = {'Majors','Interests','Group','Personal'};
record = ESM(:,6:8);
place = record{:,1};
people_str = record{:,2};
activity = record{:,3};
spaceExpr = '\d';
a = {};
for i = 1:length(people_str)
    temp = people_str{i};
    a = [a,strsplit(temp,'|')];
end
people = zeros(length(a),1);
for i = 1:length(a)
    people(i) = str2double(a{i});
end
[count1,edges1] = histcounts(place);
[count2,edges2] = histcounts(people);
[count3,edges3] = histcounts(activity);
center1 = 0.5*(edges1(1:end-1)+edges1(2:end));
center2 = 0.5*(edges2(1:end-1)+edges2(2:end));
center3 = [10,20,30,40];
subplot(2,2,[1,2]);
%
bar(center1, count1, 0.7);
set(gca,'xticklabel',label1)
ylabel('Events count');
title('place');
subplot(223);
bar(center2, count2, 0.7);
set(gca,'xticklabel',label2)
ylabel('Events count');
title('people');
subplot(224);

axis([9.5 40.5 0 2100]);
bar(center3, count3, 0.5);
set(gca,'xticklabel',label3)
ylabel('Events count');
title('activity');
