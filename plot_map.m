function plot_map(map, color)
    reward_symbols = '0123456789$';
    empty_symbol = '.';
    wall_symbol = '#';
    agent_symbol = 'X';

    I = double(map ~= wall_symbol);
    I = repmat(I, [1 1 3]);
    grey = 0.3;
    if isequal(color, 'red')
        I(:,:,1) = I(:,:,1) * (1 - grey) + grey; % R
        I(:,:,2) = grey; % G
        I(:,:,3) = grey; % B
    elseif isequal(color, 'green')
        I(:,:,1) = grey; % R
        I(:,:,2) = I(:,:,2) * (1 - grey) + grey; % G
        I(:,:,3) = grey; % B
    elseif isequal(color, 'blue')
        I(:,:,1) = grey; % R
        I(:,:,2) = grey; % G
        I(:,:,3) = I(:,:,3) * (1 - grey) + grey; % B
    else
        I(:,:,1) = I(:,:,1) * (1 - grey) + grey; % B; % R
        I(:,:,2) = I(:,:,1); % G
        I(:,:,3) = I(:,:,1); % B
    end

    image(I);
    axis off;


    font_size = 20;
    gap_size = 0.25;
    line_width = 5;

    % agent
    [ax, ay] = ind2sub(size(map), find(map == agent_symbol));
    text(ay - gap_size, ax, 'X', 'FontSize', font_size, 'FontWeight', 'bold', 'Color', 'black');

    % rewards 
    [rxs, rys] = ind2sub(size(map), find(ismember(map, reward_symbols)));
    for i = 1:numel(rxs)
        rx = rxs(i);
        ry = rys(i);
        money = ['$', map(rx, ry)];
        text(ry - gap_size, rx, money, 'FontSize', font_size * 0.5, 'FontWeight', 'bold', 'Color', 'green');
    end


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
