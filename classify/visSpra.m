
%%
%�������ص�Ԫ������ĸ���
ex1 = batchData(1,:);
hiddenB = hidBias;
w = weight;
visualB = visBias;

actPro = 1./(1+exp(-(ex1*w + hiddenB)));

%%������ͼ

ex1 = batchData(1,:);
hiddenB = hidBias;
w = weight;
visualB = visBias;

actPro = 1./(1+exp(-ex1*w + hiddenB));

%%������ͼ

x = 1:1:500;
y = actPro;

bar(x,y)
axis([0,500,0,1])
xlabel('hiddle units')
ylabel('Activation probability')

