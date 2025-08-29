%% demo for figure 5
%% preprocessing

clear;clc;
ESM_path = '../Psychol_Rec/ESM.xlsx';

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


%% correlation
clear;clc;
load('ESM_emo');
origin_label = {'upset','hostile','alert','ashamed','inspired','nervous','determined','attentive','afraid','active'};
emotion_label = {'upset','hostile','ashamed','nervous','afraid','alert','inspired','determined','attentive','active'};
trans_label = [1,2,4,6,9,3,5,7,8,10]; % let PA in first 5, NA in last 5
R_sub = zeros(142,10,10);
P_sub = zeros(142,10,10);
count = 0;
for subno = 1:142
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
title('correlation r');
colorbar;
subplot(122);
z_r = 0.5*log((1+R_sub)./(1-R_sub));
for i = 1:10
    for j = 1:10
        [~,p_z(i,j),~,info{i,j}] = ttest(z_r(1:count,i,j),0,'Alpha',0.05);
    end
end
p_z(logical(eye(size(p_z)))) = 1;
heatmap(emotion_label,emotion_label,-log10(p_z));
title('-log p');
colorbar;

figure;
histogram(R_sub(:,4,5),[-0.2:0.1:1],'FaceColor',[0.14,0.42,0.71]);
xlabel('r')
ylabel('Count');
title('Distribution of r');
%%
R_sub = zeros(142,1);
P_sub = zeros(142,1);
% count = 1;
figure;
for subno = 18%:144
    temp = data{subno};
    temp2 = temp(:,trans_label);
    [r_s, p_s] = corrcoef(temp2(:,4),temp2(:,5));
    plot(temp2(:,4),'LineWidth',1.5);
    hold on;
    plot(temp2(:,5),'LineWidth',1.5);
    xlabel('Ratings');
    ylabel('Scores');
    title('Participant 1019');
    legend('nervous','afraid');
    axis([0,30,0.5,5.5]);
    text(3,5,['r = ',num2str(r_s(1,2))]);
    text(3,4.5,['p = ',num2str(p_s(1,2))]);
%     count = count + 1;
end