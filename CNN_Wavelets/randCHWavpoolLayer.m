classdef randCHWavpoolLayer < nnet.layer.Layer % & nnet.layer.Formattable (Optional) 

    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end

    properties (Learnable)
        % Layer learnable parameters.

        %Weights
    end
    
    methods
        function layer = randCHWavpoolLayer(name)
            % This function performs the Lift Wavelet Transform
            
            % Set number of inputs.
            layer.NumInputs = 3;
            
            % Set number of outputs.
            layer.NumOutputs = 1;
            % layer.OutputNames = {'out1','out2'};
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Multiply Input by Scalar";
            
            % Initialize layer weights.
            %layer.Weights = 5*rand; %rand(224,224,numInputs);
        end
        
        function [Z1] = predict(layer,X1,X2,X3)
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
            IN2 = X2;
            IN3 = X3;
            L=randi(3,1);
            if L==1
                OUT=IN1;
            elseif L==2
                OUT=IN2;
            else
                OUT=IN3;
            end
            Z1=OUT;
        end        
        
    end
end