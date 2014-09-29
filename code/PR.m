function [rec,prec,ap] = PR(scores, gt, cli, ci)

draw = true;

[so,si]=sort(-scores);
tp=gt(si)>0;
fp=gt(si)<0;

fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/sum(gt>0);
prec=tp./(fp+tp);

% compute average precision

ap=0;
for t=0:0.1:1
    p=max(prec(rec>=t));
    if isempty(p)
        p=0;
    end
    ap=ap+p/11;
end



if draw
    % plot precision/recall
    h = plot(rec,prec,'-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(['AP= ' num2str(ap)]);
	saveas(h, ['../plot/',num2str(cli),'_',num2str(ci)], 'png');
end

