function [data_combined] = timematch(pathphy, pathesm, timeLen)
% This file is part of the primary analysis process of DREAM database.
% pathphy = the absolute path of the filefolder to which all daily
% physiological data is stored. Files in this folder are organized by
% person.
% pathesm = the absolute path of the excel file to which all ESM
%  data is stored. ESM data is in the first sheet of this file.
% timeLen = the length of physiological data that should be cut out before
% the matched ESM sampling time point. This variable is in minutes.
% For more detailed information of the data storage format, see the
% description file.
tic
subdir = dir(pathphy);
[data_esm, t_esm, row_esm] = xlsread(pathesm, 1);
data_combined = zeros(size(data_esm, 1), size(data_esm,2) + timeLen*60*2);
for i = 1 : length( subdir)
    if( isequal( subdir( i ).name, '.' )||...
            isequal( subdir( i ).name, '..')||...
            ~subdir( i ).isdir)
        continue;
    else
        ID = str2num(subdir(i).name(isstrprop(subdir(i).name, 'digit')));                       %the name of filefolder, i.e. participant ID
        id_index = find(data_esm(:, 1) == ID);
        time_of_id = datenum(row_esm(id_index+1, 2));                                                %find all esm records of the participant
        csv_in_subdirpath = fullfile( pathphy, subdir(i).name, '*.csv');
        csv_in_subdir = dir(csv_in_subdirpath);                                                                % list all csv files
        if ~isempty(csv_in_subdir)
            csvpath = fullfile(pathphy, subdir(i).name, '*.csv');
            csvdir = dir(csvpath);
            for cnt = 1 : length(csvdir)
                if length(csvdir(cnt).name)==33                                                                    %csv with summary data
                    csvname_temp = (fullfile(csvdir(cnt).folder, csvdir(cnt).name))
                    [data_phy, t_phy, row_phy] = xlsread(csvname_temp, 1);
                    timestr = string(row_phy(2: end, 5));
                    timenum = datenum(timestr);
                    uplim =  max(timenum);
                    lowlim = min(timenum);
                    for k = 1 : length(id_index)                                                                          %for all esm records of the participant
                        data_combined(id_index(k), 1: size(data_esm,2)) = data_esm(id_index(k), :);
                        if (time_of_id(k)>lowlim) && (time_of_id(k)<uplim)                           %find the matched time point in physiological data
                            matchpoint = find(timenum == time_of_id(k));
                            if length(matchpoint) ~= 1
                                ID
                                datestr(time_of_id(k))
                                length(matchpoint)
                            else
                                if matchpoint>timeLen*60                                                              % if the matched time point is found and the timeLen is coverd in the current physiological data
                                    hr_match = data_phy((matchpoint - timeLen*60 + 1) : matchpoint, 1);
                                    gsr_match = data_phy((matchpoint - timeLen*60 + 1) : matchpoint, 3);
                                    data_combined(id_index(k), :) = [data_esm(id_index(k), :) hr_match' gsr_match'];
                                else
                                    ID
                                    datestr(time_of_id(k))
                                    matchpoint;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
toc