
[starting time]_[ending time].csv : preprocessed data
	In preprocessed data file, all data were formatted at a 1-Hz sampling rate for the convenience of further analysis. 
	Specifically, the heart rate was computed at a 1-Hz sampling rate with PPG data from a preceding 10-sec time window. The computation was implemented in the software package provided by the manufacturer of the wristband, featured by a joint sparse spectrum reconstruction algorithm to remove motion-related artifacts32. 
	The GSR data were simply downsampled from the original 40-Hz raw data, and the motion data were computed as the downsampled root mean square of the 20-Hz three-axis raw acceleration data to reflect the overall motion intensity. 
	*Note that the minimal output value of heart rate value from the device was 40. The value of 40 was obtained when 1) no valid heart rate could be computed due to unreliable PPG signal acquisition; 2) the computed heart rate value was less than 40. 
	In addition, the minimal output value of GSR from the device was 0.00624 μS, indicating high skin impedance that could be caused by relatively loose skin contact. As these data could be meaningful (although likely to be missing values), these values were kept in the data file and the users are suggested to interpret them with caution.
	

[starting time]_[ending time]_PPG.csv : raw data at the original sampling rate(20Hz) for PPG

[starting time]_[ending time]_GSR.csv : raw data at the original sampling rate(40Hz) for GSR signal

[starting time]_[ending time]_ACC.csv : raw data at the original sampling rate(20Hz) for the three-axis acceleration