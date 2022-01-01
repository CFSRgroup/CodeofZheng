fop = fopen('eeg.txt', 'wt' );%C.txt是即将保存的txt文件
load D:\USST_work\matlab\trial_procedure\study_1\ex1_filter_ica\session1\s1_1.mat
[M,N] = size(eeg);
for m = 1:M
for n = 1:N
fprintf( fop,' %s', mat2str(eeg(m,n)));%AA是matlab矩阵
end
fprintf(fop, '\n' );
end