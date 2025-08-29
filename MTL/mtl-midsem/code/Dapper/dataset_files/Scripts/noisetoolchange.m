 
function [od, ow]= noisetoolchange(input_data,degree1, degree2)
format long e;

[~,w] = nt_detrend(input_data',degree1);

[output_data.data,~,~] = nt_detrend(input_data',degree2); 
output_data.w = w;

ow = output_data.w;
od = output_data.data;
