classdef LayerOddV < nnet.layer.Layer % & nnet.layer.Formattable (Optional) 

    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end

    properties (Learnable)
        % Layer learnable parameters.

        %Weights
    end
    
    methods
        function layer = LayerOddV(name)
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
            IN1(:,2,:,:)=IN1(:,2,:,:)*0;
            IN1(:,4,:,:)=IN1(:,4,:,:)*0;
            IN1(:,6,:,:)=IN1(:,6,:,:)*0;
            IN1(:,8,:,:)=IN1(:,8,:,:)*0;
            IN1(:,10,:,:)=IN1(:,10,:,:)*0;
            IN1(:,12,:,:)=IN1(:,12,:,:)*0;
            IN1(:,14,:,:)=IN1(:,14,:,:)*0;
            IN1(:,16,:,:)=IN1(:,16,:,:)*0;
            IN1(:,18,:,:)=IN1(:,18,:,:)*0;
            IN1(:,20,:,:)=IN1(:,20,:,:)*0;
            IN1(:,22,:,:)=IN1(:,22,:,:)*0;
            IN1(:,24,:,:)=IN1(:,24,:,:)*0;
            Z1=IN1;
        end        
        
    end
end