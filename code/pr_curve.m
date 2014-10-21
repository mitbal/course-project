function pr_curve( rec1, prec1, rec2, prec2, plot_title )

    is_save = true;

    h = figure;
    hold on;
    plot(rec1, prec1, 'b-');
    plot(rec2, prec2, 'r-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(plot_title);
    hold off;

    if is_save
        saveas(h, ['../plot/', plot_title], 'png');
    end
end
