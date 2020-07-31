function  auc = plot_roc( predict, ground_truth,color )  
% INPUTS  
%  predict       - �������Բ��Լ��ķ�����  
%  ground_truth - ���Լ�����ȷ��ǩ,����ֻ���Ƕ����࣬��0��1  
% OUTPUTS  
%  auc            - ����ROC���ߵ������µ����  
  
%��ʼ��Ϊ��1.0, 1.0��  
x = 1.0;  
y = 1.0;  
%�����ground_truth������������Ŀpos_num�͸���������Ŀneg_num  
pos_num = sum(ground_truth==1);  
neg_num = sum(ground_truth==0);  
%���ݸ���Ŀ���Լ������x�����y��Ĳ���  
x_step = 1.0/neg_num;  
y_step = 1.0/pos_num;  
%���ȶ�predict�еķ��������ֵ���մ�С��������  
[predict,index] = sort(predict);  
ground_truth = ground_truth(index);  
%��predict�е�ÿ�������ֱ��ж�������FP������TP  
%����ground_truth��Ԫ�أ�  
%��ground_truth[i]=1,��TP������1����y�᷽���½�y_step  
%��ground_truth[i]=0,��FP������1����x�᷽���½�x_step  
for i=1:length(ground_truth)  
    if ground_truth(i) == 1  
        y = y - y_step;  
    else  
        x = x - x_step;  
    end  
    X(i)=x;  
    Y(i)=y;  
end  
%����ͼ��       
%plot(X,Y,'-go','LineWidth',1,'MarkerSize',2);  
plot(X,Y,color,'LineWidth',1,'MarkerSize',2);  
xlabel('False Positive Rate');  
ylabel('True Positive Rate');  
title('ROC');  
%����С���ε����,����auc  
auc = -trapz(X,Y)            
end 