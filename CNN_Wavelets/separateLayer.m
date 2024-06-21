classdef separateLayer < nnet.layer.Layer % & nnet.layer.Formattable (Optional) 

    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end

    properties (Learnable)
        % Layer learnable parameters.

        %Weights
    end
    
    methods
        function layer = separateLayer(name)
            % This function performs the Lift Wavelet Transform
            
            % Set number of inputs.
            layer.NumInputs = 1;
            
            % Set number of outputs.
            layer.NumOutputs = 4;
            % layer.OutputNames = {'out1','out2'};
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Separate elements in positions (1,1),(1,2),(2,1)&(2,2) in 2x2 regions";
            
            % Initialize layer weights.
            %layer.Weights = 5*rand; %rand(224,224,numInputs);
        end
        
        function [Z1,Z2,Z3,Z4] = predict(layer,X1)
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
            Z1 = X1;
            Z2 = X1;
            Z3 = X1;
            Z4 = X1;
            M1 = IN1;
            M2 = IN1;
            M3 = IN1;
            M4 = IN1;
            Pattern1 = IN1;
            Pattern2 = IN1;
            Pattern3 = IN1;
            Pattern4 = IN1;
            %[x,y] = size(IN1);
            Pattern1 = [1,0;0,0];
            Pattern2 = [0,1;0,0];
            Pattern3 = [0,0;1,0];
            Pattern4 = [0,0;0,1];
            M1 = repmat(Pattern1,size(IN1)/2); % Pattern,x/2,y/2
            M2 = repmat(Pattern2,size(IN1)/2);
            M3 = repmat(Pattern3,size(IN1)/2);
            M4 = repmat(Pattern4,size(IN1)/2);
            %%%
            Z1=IN1.*M1;
            Z2=IN1.*M2;
            Z3=IN1.*M3;
            Z4=IN1.*M4;
        end        
        
    end
end