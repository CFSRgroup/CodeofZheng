function [hr_new]=Data_process_outlier_hr(hr,k)
%����������������Ⱥ���޳��Ͳ���,k��ʾ��׼��ı���,����ȡ3
%���þ�ֵ����
[p q]=find(hr==0);
for i0=1:length(p)
    hr(p(i0))=mean(hr);
end
%����sigmak
sigmak=k;
%�ж��Ƿ���ڴִ����
if max(zscore(hr))>=sigmak | min(zscore(hr))<=-sigmak
    %��������,�����һάK��ֵ����
   [model,data.y]=kmeans(hr,2);
   %�ҳ�ģ���������
   [a b]=find(model==1);
   [c d]=find(model==2);
   [e f]=find(zscore(hr)>=(sigmak) | zscore(hr)<=(-sigmak));
    m=size(a);
    n=size(c);
    hr_new=hr;
    %����һ��������Ŀ��,����Ϊ��Ϊ��������
    if m(1)>n(1)
        sum_clear=hr(a(1));
        for i1=1:length(a)-1
            sum_clear=sum_clear+hr(a(i1+1));
        end
        %�����������ݾ�ֵ
        ava_clear=sum_clear/length(a);
        %���쳣������滻
        for i2=1:length(e)
            hr_new(e(i2))=ava_clear;
        end
    else if m(1)<=n(1)
        sum_clear=hr(c(1));
        for i1=1:length(c)-1
            sum_clear=sum_clear+hr(c(i1+1));
        end
        ava_clear=sum_clear/length(c);
        for i2=1:length(e)
            hr_new(e(i2))=ava_clear;
        end
        else
        hr_new=hr;    
        end
    end
else
    hr_new=hr;
end
    
        