function [sen, spec] = sen_spec(predict, label)


pre = sign(predict);
pre(find(pre==-1)) = 0;

TP = 0;
TN = 0;
FN = 0;
FP = 0;

len = length(predict);


for i = 1: len
    if(pre(i) == 1 && label(i) == 1)
        TP = TP + 1;
    end
    if(pre(i) == 0 && label(i) == 1)
        FN = FN + 1;
    end
    if(pre(i) == 0 && label(i) == 0)
        TN = TN + 1;
    end
    if(pre(i) == 1 && label(i) == 0)
        FP = FP + 1;
    end
 
end

sen = TP/(TP+FN);
spec = TN/(FP+TN);

end
