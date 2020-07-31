
%%
%计算隐藏单元被激活的概率
ex1 = batchData(1,:);
hiddenB = hidBias;
w = weight;
visualB = visBias;

actPro = 1./(1+exp(-(ex1*w + hiddenB)));

%%画柱形图

ex1 = batchData(1,:);
hiddenB = hidBias;
w = weight;
visualB = visBias;

actPro = 1./(1+exp(-ex1*w + hiddenB));

%%画柱形图

x = 1:1:500;
y = actPro;

bar(x,y)
axis([0,500,0,1])
xlabel('hiddle units')
ylabel('Activation probability')

