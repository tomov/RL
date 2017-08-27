% Options framework as in Sutton et al (1999)
%
classdef Options < handle

    properties (Constant = true)
        % Maze
        %
        pseudoreward_symbol = '$';
        subtask_symbol = 'S';
    end

    properties (Access = public)
    end

    methods

        % Initialize an Options framework from a maze
        %
        function self = Options(map)
			subtask_inds = find(map == Options.subtask_symbol)';
			map(subtask_inds) = TD.empty_symbol;

			for s = subtask_inds
                map(s) = Options.pseudoreward_symbol;

                T = TD(map);
                for i = 1:50
                    T.sampleQ();
                end

                map(s) = Options.empty_symbol;
			end

        end 
    end
end
