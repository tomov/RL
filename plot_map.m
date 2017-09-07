function plot_map(map)
    reward_symbol = '$';
    empty_symbol = '.';
    wall_symbol = '#';
    agent_symbol = 'X';
    imagesc(map == wall_symbol);
    font_size = 40;
    gap_size = 0.25;
    line_width = 5;

    % agent
    [ax, ay] = ind2sub(size(map), find(map == agent_symbol));
    text(ay - gap_size, ax, 'X', 'FontSize', font_size, 'FontWeight', 'bold', 'Color', 'red');

    % rewards 
    %[rx, ry] = ind2sub(size(map), find(map == reward_symbol));
    %text(ry - gap_size, rx, '$', 'FontSize', font_size, 'FontWeight', 'bold', 'Color', 'green');


    % path
    %{
    hold on;
    dy = ry - ay;
    dx = rx - ax;
    quiver(ay + sign(dy) * gap_size * 1.5, ax + sign(dx) * gap_size * 1.5, dy - sign(dy) * gap_size * 3, dx - sign(dx) * gap_size * 3, 0, 'Color', 'cyan', 'LineWidth', line_width);
    hold off;
    %}

    axis off;
end
