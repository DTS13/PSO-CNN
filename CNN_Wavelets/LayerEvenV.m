classdef LayerEvenV < nnet.layer.Layer % & nnet.layer.Formattable (Optional) 

    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end

    properties (Learnable)
        % Layer learnable parameters.

        %Weights
    end
    
    methods
        function layer = LayerEvenV(name)
            % This function performs the Lift Wavelet Transform
            
            % Set number of inputs.
            layer.NumInputs = 1;
            
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
        
        function [Z1] = predict(layer,X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            %X = varargin;
            %W = layer.Weights;
            
            % Initialize output
            IN1 = X; %{1};
            IN1(:,1,:,:)=IN1(:,1,:,:)*0;
            IN1(:,3,:,:)=IN1(:,3,:,:)*0;
            IN1(:,5,:,:)=IN1(:,5,:,:)*0;
            IN1(:,7,:,:)=IN1(:,7,:,:)*0;
            IN1(:,9,:,:)=IN1(:,9,:,:)*0;
            IN1(:,11,:,:)=IN1(:,11,:,:)*0;
            IN1(:,13,:,:)=IN1(:,13,:,:)*0;
            IN1(:,15,:,:)=IN1(:,15,:,:)*0;
            IN1(:,17,:,:)=IN1(:,17,:,:)*0;
            IN1(:,19,:,:)=IN1(:,19,:,:)*0;
            IN1(:,21,:,:)=IN1(:,21,:,:)*0;
            IN1(:,23,:,:)=IN1(:,23,:,:)*0;
            Z1=IN1;
        end        
        
    end
end