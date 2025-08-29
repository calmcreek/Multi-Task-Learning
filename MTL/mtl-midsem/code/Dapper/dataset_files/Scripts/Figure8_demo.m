%% demo for figure 8
%% Time match and feature extraction
% Time match may cost a relatively long time

pathphy = '../Physiol_Rec';
pathesm = '../Psychol_Rec/ESM.xlsx';

timeLen = 30;
data_combined = timematch(pathphy, pathesm, timeLen);
itemnum_of_esm = 26;
Wid_in = timeLen*60;
Wid_out = 30*60;
% extract features using NoiseTools
% Artifacts are removed. 
[data_final, type] = extractfeatures(data_combined, itemnum_of_esm, Wid_in, Wid_out);

save('match_res','data_final');

%% Correlation between Phychol assessments and Phisiol features

clc;
load match_res.mat;
sublist = unique(data_final(:,1));
subnum = length(sublist);
elenum = zeros(subnum,1);
r_sub = zeros(subnum,10,4);
p_sub = zeros(subnum,10,4);
origin_label = {'upset','hostile','alert','ashamed','inspired','nervous','determined','attentive','afraid','active'};
emotion_label = {'upset','hostile','ashamed','nervous','afraid','alert','inspired','determined','attentive','active'};
physio_label = {'medHR','quarHR','medGSR','quarGSR'};
trans_label = [1,2,4,6,9,3,5,7,8,10]; % let PA in first 5, NA in last 5
count = 0;
for i = 1:subnum
    target = find(data_final(:,1) == sublist(i));
    elenum(i) = length(target);
    emo = data_final(target,15:24);
    physio = data_final(target,27:30);
    if length(emo) < 3
        continue;
    end
    emo = emo(:,trans_label);
    [r,p] = intercorr(emo,physio);
    if isequal(ismissing(r),zeros(10,4))
        count = count + 1;
        r_sub(i,:,:) = r;
        p_sub(i,:,:) = p;
    end
end

R = squeeze(sum(r_sub,1)/count);
P = squeeze(sum(p_sub,1)/count);
figure;
subplot(121);
heatmap(physio_label,emotion_label,R);
% set(gca,'XTickLabel',physio_label);
% set(gca,'YTickLabel',emotion_label,'FontSize',8);
title('correlation r');
colorbar;
subplot(122);
z_r = 0.5*log((1+r_sub)./(1-r_sub));
for i = 1:10
    for j = 1:4
        [~,p_z(i,j),~,info{i,j}] = ttest(z_r(1:count,i,j),0,0.05);
    end
end

heatmap(physio_label,emotion_label,-log(p_z));
title('-log p');
% set(gca,'XTickLabel',physio_label);
% set(gca,'YTickLabel',emotion_label,'FontSize',8);
colorbar;

figure;
histogram(r_sub(:,5,2),'FaceColor',[0.14,0.42,0.71]);
xlabel('r');
ylabel('Count');
title('Distribution of r');

figure;
temp = squeeze(r_sub(:,5,2));
a = find(temp < -0.2);
b = find(temp > -0.2&temp < 0.2);
c = find(temp > 0.2);
k = 1;
for i = [6,8]
    subplot(1,2,k);
    target = find(data_final(:,1) == sublist(c(i)));
    emo = data_final(target,15:24);
    physio = data_final(target,27:30);
    plot(physio(:,2),emo(:,5),'k*');
    axis([0,30,0,5]);
    xlabel('QuarHR');
    ylabel('Inspired');
    new = temp(c(i));
    text(15,4.5,['r = ',num2str(roundn(new,-2))]);
    k = k+1;
end

function [r,p] = intercorr(a,b)
    num1 = size(a,2);
    num2 = size(b,2);
    r = zeros(num1,num2);
    p = zeros(num1,num2);
    for i = 1:num1
        for j = 1:num2
            [r(i,j), p(i,j)] = corr(a(:,i),b(:,j),'Type','Spearman');
        end
    end

end