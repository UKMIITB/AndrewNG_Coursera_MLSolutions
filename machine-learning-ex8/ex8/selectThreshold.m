function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    tp=0;fp=0;fn=0;y=0;
    for i=1:size(yval,1)
        if pval(i)<epsilon
            y=1;
        end
        if (yval(i)==1) & (y==1)
            tp=tp+1;
        end
         if (yval(i)==0) & (y==1)
             fp=fp+1;
         end
         if (yval(i)==1) & (y==0)
             fn=fn+1;
         end
    end
    prec=tp/(tp+fp);
    rec=tp/(tp+fn);
    F1=(2*prec*rec)/(prec+rec);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
