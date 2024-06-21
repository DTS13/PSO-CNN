classdef stochpoolLayer < nnet.layer.Layer % & nnet.layer.Formattable (Optional) 

    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end

    properties (Learnable)
        % Layer learnable parameters.

        %Weights
    end
    
    methods
        function layer = stochpoolLayer(name)
            % This function performs the Lift Wavelet Transform
            
            % Set number of inputs.
            layer.NumInputs = 4;
            
            % Set number of outputs.
            layer.NumOutputs = 1;
            % layer.OutputNames = {'out1','out2'};
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Stochastic Pooling";
            
            % Initialize layer weights.
            %layer.Weights = 5*rand; %rand(224,224,numInputs);
        end
        
        function [Z1] = predict(layer,X1,X2,X3,X4)
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
            IN4 = X4;
            OUT = IN1;
            P1 = IN1;
            P2 = IN2;
            P3 = IN3;
            P4 = IN4;
            SUM1 = IN1;
            Z1 = IN1;
            %%%%
            SUM1 = IN1 + IN2 + IN3 + IN4;
            P1 = (IN1./SUM1);
            P2 = (IN2./SUM1);
            P3 = (IN3./SUM1);
            P4 = (IN4./SUM1);         
            P = cat(3,P1,P2,P3,P4); 
            S = cat(3,IN1,IN2,IN3,IN4);
            randsample(S,OUT,true,P) % solo vectores
            
            OUT5 = cat(3,OUT1,OUT2,OUT3,OUT4);
            OUT5 = sort(OUT5,3);
            OUT7 = cat(3,IN1,IN2,IN3,IN4);
            OUT8 = sort(OUT7,3);
            OUT5(:,:,3) = OUT5(:,:,3) + OUT5(:,:,4);
            OUT5(:,:,2) = OUT5(:,:,2) + OUT5(:,:,3) + OUT5(:,:,4);
            OUT5(:,:,1) = OUT5(:,:,1) + OUT5(:,:,2) + OUT5(:,:,3) + OUT5(:,:,4);
            L=rand(size(IN1));
            OUT6(:,:,1)=OUT5(:,:,1)-L;
            OUT6(:,:,2)=OUT5(:,:,2)-L;
            OUT6(:,:,3)=OUT5(:,:,3)-L;
            OUT6(:,:,4)=OUT5(:,:,4)-L;
            MIN = min(OUT6,[],3);
            OUT5(:,:,1)=OUT6(:,:,1)-MIN;
            OUT5(:,:,2)=OUT6(:,:,2)-MIN;
            OUT5(:,:,3)=OUT6(:,:,3)-MIN;
            OUT5(:,:,4)=OUT6(:,:,4)-MIN;
            OUT6 = round(exp(-abs(OUT5*16)),0);
            OUT7 = OUT8.*OUT6;
            OUT = OUT7(:,:,1) + OUT7(:,:,2) + OUT7(:,:,3) + OUT7(:,:,4);
            Z1=OUT;
        end        
        
    end
end