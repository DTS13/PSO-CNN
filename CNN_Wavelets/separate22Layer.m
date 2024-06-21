classdef separate22Layer < nnet.layer.Layer % & nnet.layer.Formattable (Optional) 

    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end

    properties (Learnable)
        % Layer learnable parameters.

        %Weights
    end
    
    methods
        function layer = separate22Layer(name)
            % This function performs the Lift Wavelet Transform
            
            % Set number of inputs.
            layer.NumInputs = 1;
            
            % Set number of outputs.
            layer.NumOutputs = 1;
            % layer.OutputNames = {'out1','out2'};
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Separate elements in position (2,2) in 2x2 regions";
            
            % Initialize layer weights.
            %layer.Weights = 5*rand; %rand(224,224,numInputs);
        end
        
        function [Z1] = predict(layer,X1)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            %X = varargin;
            %W = layer.Weights;
            
            % Initialize output
            IN1 = X1;
            Z1 = IN1;
            M1 = IN1;
            %[x,y] = size(IN1);
            Pattern = [0,0;0,1];
            M1 = repmat(Pattern,size(IN1,1)/2,size(IN1,2)/2,size(IN1,3)); % Pattern,x/2,y/2
            %%%
            Z1=IN1.*M1;
        end        
        
    end
end