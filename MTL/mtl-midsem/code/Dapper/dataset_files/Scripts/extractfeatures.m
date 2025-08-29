function [data_final, type] = extractfeatures(data_in, itemnum_of_esm, Wid_in, Wid_out)
% This file is part of the primary analysis process of DREAM database.
% extractfeatures eliminate outliers in physiological data, and extract
% 6 features for further analysis: median of HR, interquartile of HR,
% median of low frequency gsr, interquartile of low frequency gsr,
% median of high frequency gsr, interquartile of high frequency gsr.
% data_in = the combined data of ESM and physiological records.
% itemnum_of_esm = the number of items in each ESM records.
% Wid_in = the length of physiological data = timeLen*60.
% Wid_out = the length of physiological data used when extracting features.
% For more detailed information of the data storage format, see the
% description file.

WL_hr = 900;
WL_gsr = 600;                                                                 % the length of time window used when filting data with noisetool
data_phy = data_in(:, itemnum_of_esm+1:end);
data_phy_new = zeros(size(data_phy, 1), 5*Wid_in);
for i = 1 : size(data_phy, 1)
    hr_o = data_phy(i, 1: Wid_in);
    gsr_o = data_phy(i, Wid_in+1: 2*Wid_in);
    if (sum(gsr_o<0.01)<1700) && (var(hr_o)>1e-10)            %if there are signals of both gsr and hr, i.e. if the wristband was worn
        for j = 1 : Wid_in/WL_hr
            if var(hr_o(1+(j-1)*WL_hr : j*WL_hr))>1e-10              % if there are signals of hr during this time window, use noisetool to label outliers
                [hr_n.data(1+(j-1)*WL_hr : j*WL_hr), hr_n.w(1+(j-1)*WL_hr : j*WL_hr)] = noisetoolchange(hr_o(1+(j-1)*WL_hr : j*WL_hr), 10, 30);
            else
                hr_n.data(1+(j-1)*WL_hr : j*WL_hr) = zeros(WL_hr, 1);
                hr_n.w(1+(j-1)*WL_hr : j*WL_hr) = zeros(WL_hr, 1);
            end
        end
        for j = 1 : Wid_in/WL_gsr
            if sum(gsr_o(1+(j-1)*WL_gsr : j*WL_gsr) < 0.01) < WL_gsr*0.9  % if there are more than 10% signals in this time window, , use noisetool to label outliers
                [gsr_n.data(1+(j-1)*WL_gsr :j*WL_gsr),gsr_n.w(1+(j-1)*WL_gsr:j*WL_gsr)] = noisetoolchange(gsr_o(1+(j-1)*WL_gsr : j*WL_gsr), 10, 30);
            else
                gsr_n.data(1+(j-1)*WL_gsr:j*WL_gsr) = zeros(WL_gsr, 1);
                gsr_n.w(1+(j-1)*WL_gsr:j*WL_gsr) = zeros(WL_gsr, 1);
            end
        end
        if sum((gsr_o < 0.01) | (gsr_n.w < 1)) < Wid_in/2                                  % if both gsr data and hr data are of proper quality     if sum((hr_o < 42) | (hr_n.w < 1)) < Wid_in/2
            if sum((hr_o <42) | (hr_n.w<1)) < Wid_in/2
                gsr_n.w(find(gsr_o < 0.01)) = 0;                                                         %make sure all outliers are labeled
                hr_n.w(find(hr_o < 42)) = 0;
                data_phy_new(i, :) = [hr_o gsr_n.data gsr_o hr_n.w gsr_n.w];
            end
        end
    end
end
data_combinetemp = [data_in(:, 1: (itemnum_of_esm)), data_phy_new];
data_combine = data_combinetemp(any(data_combinetemp(:, itemnum_of_esm+1+3*Wid_in : itemnum_of_esm+Wid_in*5), 2), :);
if mod(Wid_in, Wid_out)==0
    type = 'multiple';
    data_final = zeros(size(data_combine, 1), itemnum_of_esm+6, Wid_in/Wid_out);
    for i = 1 : Wid_in/Wid_out
        left =  Wid_out*(i-1)+itemnum_of_esm+1;
        right = Wid_out*i + itemnum_of_esm;
        hr_temp = data_combine(:, left: right);
        gsr_temp_high = data_combine(:, Wid_in+ left : Wid_in+ right);
        gsr_temp_low = data_combine(:, 2*Wid_in +left : 2*Wid_in + right) - gsr_temp_high;
        hr_label = data_combine(:, 3*Wid_in+left : 3*Wid_in+right);
        gsr_label = data_combine(:, 4*Wid_in+left:4*Wid_in+right);
        for cnt = 1 : size(data_combine, 1)
            hr_labeled = hr_temp(cnt, find(hr_label(cnt, :)>0));
            gsr_labeled_high = gsr_temp_high(cnt, find(gsr_label(cnt, :)>0));
            gsr_labeled_low = gsr_temp_low(cnt, find(gsr_label(cnt, :)>0));
            % HR features: median of HR, interquartile of HR,
            hr_feature = [prctile(hr_labeled, 50), prctile(hr_labeled, 75)-prctile(hr_labeled, 25)];
            % GSR features: median of low frequency gsr, interquartile of low frequency gsr,
            % median of high frequency gsr, interquartile of high frequency gsr.
            gsr_feature = [prctile(gsr_labeled_low, 50), (prctile(gsr_labeled_low, 75)-prctile(gsr_labeled_low, 25)),prctile(gsr_labeled_high, 50), (prctile(gsr_labeled_high, 75)-prctile(gsr_labeled_high, 25))];
            data_final(cnt, : , i) = [data_combine(cnt, 1: itemnum_of_esm), hr_feature, gsr_feature];
        end
    end
else
    type = 'single';
    data_final = zeros(size(data_combine, 1), itemnum_of_esm+6);
    
    left =  Wid_in - Wid_out + itemnum_of_esm+1;
    right = Wid_in+ itemnum_of_esm;
    hr_temp = data_combine(:, left: right);
    gsr_temp_high = data_combine(:, Wid_in+ left : Wid_in+ right);
    gsr_temp_low = data_combine(:, 2*Wid_in +left : 2*Wid_in + right) - gsr_temp_high;
    hr_label = data_combine(:, 3*Wid_in+left : 3*Wid_in+right);
    gsr_label = data_combine(:, 4*Wid_in+left:4*Wid_in+right);
    for cnt = 1 : size(data_combine, 1)
        hr_labeled = hr_temp(cnt, find(hr_label(cnt, :)>0));
        gsr_labeled_high = gsr_temp_high(cnt, find(gsr_label(cnt, :)>0));
        gsr_labeled_low = gsr_temp_low(cnt, find(gsr_label(cnt, :)>0));
        % HR features: median of HR, interquartile of HR,
        hr_feature = [prctile(hr_labeled, 50), prctile(hr_labeled, 75)-prctile(hr_labeled, 25)];
        % GSR features: median of low frequency gsr, interquartile of low frequency gsr,
        % median of high frequency gsr, interquartile of high frequency gsr.
        gsr_feature = [prctile(gsr_labeled_low, 50), (prctile(gsr_labeled_low, 75)-prctile(gsr_labeled_low, 25)),prctile(gsr_labeled_high, 50), (prctile(gsr_labeled_high, 75)-prctile(gsr_labeled_high, 25))];
        data_final(cnt, :) = [data_combine(cnt, 1: itemnum_of_esm), hr_feature, gsr_feature];
    end
    
end