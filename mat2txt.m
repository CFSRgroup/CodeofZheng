fop = fopen('eeg.txt', 'wt' );%C.txt�Ǽ��������txt�ļ�
load D:\USST_work\matlab\trial_procedure\study_1\ex1_filter_ica\session1\s1_1.mat
[M,N] = size(eeg);
for m = 1:M
for n = 1:N
fprintf( fop,' %s', mat2str(eeg(m,n)));%AA��matlab����
end
fprintf(fop, '\n' );
end