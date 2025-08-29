%% demo for figure 6
%% demo signal for ID:1001
clear;clc;
pathphy = '../Physiol_Rec';
pathesm = '../Psychol_Rec/ESM.xlsx';
subdir = dir(pathphy);

[data_esm, t_esm, row_esm] = xlsread(pathesm, 1);

for i = 3%:length(subdir)
    ID = str2num(subdir(i).name(isstrprop(subdir(i).name, 'digit'))); %the name of filefolder, i.e. participant ID
    id_index = find(data_esm(:, 1) == ID);
    time_of_id = datenum(row_esm(id_index+1, 2)); %find all esm records of the participant
    csv_in_subdir = dir(fullfile( pathphy, subdir(i).name, '*.csv'));   % list all csv files
    for cnt = 1 : length(csv_in_subdir)
        if length(csv_in_subdir(cnt).name)==33      %find *.csv
            csvname_temp = fullfile(csv_in_subdir(cnt).folder, csv_in_subdir(cnt).name);
            [data_phy, t_phy, row_phy] = xlsread(csvname_temp, 1);
            timestr = string(row_phy(2: end, 5));
            timenum = datenum(timestr);
            temp_2 =  max(timenum);
            temp_1 = min(timenum);
            temp = timestr{end};
            range_1 = datenum([temp(1:11),'9:00:00']);
            range_2 = datenum([temp(1:11),'22:00:00']);
            uplim = min(temp_2,range_2);
            lowlim = max(temp_1,range_1);
            plot_range = find(timenum>lowlim&timenum<uplim);
            colorvar = [0.8,0.16,0.65;0.33,0.31,0.60;0.16,0.53,0.26];
            titlevar = {'HR','Motion','GSR'};
            duration = 5000/3600/24;
            x_tick = datestr(lowlim:duration:uplim+duration);
            myylabel = {'bpm','m/s^2','uS'};
            
            for feat = 1:3
                subplot(3,1,feat);
                plot_y = data_phy(plot_range,feat);
                xx = 1:length(plot_y);
                if feat == 1
                    plot_y = downsample(plot_y,60);
                    xx = downsample(xx,60);
                end
                plot(xx,plot_y,'Color',colorvar(feat,:));
                hold on;
                for k = 1 : length(id_index)                                                                          %for all esm records of the participant
                    if (time_of_id(k)-1/48>lowlim) && (time_of_id(k)<uplim)                           %find the matched time point in physiological data
                        disp(datestr(time_of_id(k)));
                        plot_x = (time_of_id(k)-lowlim)*24*3600 + 1;
                        rectangle('Position',[plot_x-1800,min(plot_y),1800,max(plot_y)-min(plot_y)],...
                            'Curvature',0.8,'Edgecolor',[1,0.2,0.2],'LineWidth',1);
                    end
                end
                title(titlevar{feat});
                ylabel(myylabel{feat});
                set(gca,'XTickLabel',x_tick(:,13:17));
                if feat == 1
                    axis([0,46800,30,110]);
                end
                if feat == 2
                    axis([0,46800,5,30]);
                end
                if feat == 3
                    axis([0,46800,0,4]);
                end
            end
            break;
        end
    end
end

%% read all participants' data
clear;clc;
pathphy = '../Physiol_Rec';
subdir = dir(pathphy);
count = 1;
for i = 3:length(subdir)
    ID = str2num(subdir(i).name(isstrprop(subdir(i).name, 'digit'))); 
    disp(ID);%the name of filefolder, i.e. participant ID                                            %find all esm records of the participant
    csv_in_subdir = dir(fullfile( pathphy, subdir(i).name, '*.csv'));                                                                % list all csv files
    for cnt = 1 : length(csv_in_subdir)
        if length(csv_in_subdir(cnt).name)==33                                                                    %csv with summary data
            csvname_temp = fullfile(csv_in_subdir(cnt).folder, csv_in_subdir(cnt).name);
            [data_phy, t_phy, row_phy] = xlsread(csvname_temp, 1);
            timestr = string(row_phy(2: end, 5));
            timenum = datenum(timestr);
            temp = timestr{end};
            range_1 = datenum([temp(1:11),'9:00:00']);
            range_2 = datenum([temp(1:11),'22:00:00']);
            plot_range = find(timenum>=range_1&timenum<range_2);
            start_point = find(timenum == range_1);
            if isempty(plot_range)
                continue;
            end
            data = zeros(46800,4);
            data(:,1) = 1:46800;
            secs = 1+round((timenum(plot_range) - range_1)*3600*24);
            data(secs,2:4) = data_phy(plot_range,1:3);

            data_final{count} = data;
            count = count + 1;
        end
    end
end
save grand_data.mat data_final

%% caculate grand-average
clear;clc;
range_1 = datenum([today,'9:00:00']);
range_2 = datenum([today,'22:00:00']);
load grand_data.mat
all_data = zeros(237,46800,3);
for i = 1:size(data_final,2)
    temp = data_final{i};
    misstag = find(temp(:,2)==0&temp(:,3)==0&temp(:,4)==0);
    temp(misstag,2:4) = nan;
    all_data(i,:,:) = temp(:,2:4);
end
res = squeeze(nanmean(all_data,1));
colorvar = [0.8,0.16,0.65;0.33,0.31,0.60;0.16,0.53,0.26];
titlevar = {'HR','Motion','GSR'};
duration = 5000/3600/24;
x_tick = datestr(range_1:duration:range_2+duration);
myylabel = {'bpm','m/s^2','uS'};
for feat = 1:3
    subplot(3,1,feat);
    plot_y = res(:,feat);
    xx = 1:length(plot_y);
    plot_y = downsample(plot_y,60);
    xx = downsample(xx,60);
    plot(xx,plot_y,'Color',colorvar(feat,:));
    ylabel(myylabel{feat});
    title(titlevar{feat});
    set(gca,'XTickLabel',x_tick(:,13:17));
    if feat == 1
        axis([0,46800,55,70]);
    end
    if feat == 2
        axis([0,46800,9.6,10.2]);
    end
    if feat == 3
        axis([0,46800,0.5,1]);
    end
end

