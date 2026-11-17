import numpy as np


"""Camada densa com ReLU ou softmax."""
class Layer:
    
    def __init__(self, n_inputs, n_neurons, ativacao=None):
        if ativacao not in (None, "relu", "softmax"):
            raise ValueError("Ativação deve ser None, 'relu' ou 'softmax'.")
        
        # self.pesos começa com valores aleatórios multiplicados por 0.1
        self.pesos = 0.1 * np.random.randn(n_inputs, n_neurons)

        # self.bias começa zerado
        self.bias = np.zeros((1, n_neurons))
        self.ativacao = ativacao
        self.output = None
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs

        # self.output = inputs @ self.pesos + self.bias
        self.output = np.dot(inputs, self.pesos) + self.bias

        if self.ativacao == "relu":
            self.output = self.relu(self.output)

        elif self.ativacao == "softmax":
            self.output = self.softmax(self.output)

        return self.output

    def backward(self, gradiente, gradiente_logits=False):
        """Calcula os gradientes da camada e devolve o gradiente das entradas."""
        if self.inputs is None:
            raise RuntimeError("Execute forward antes de backward.")

        if gradiente_logits and self.ativacao != "softmax":
            raise ValueError("Gradiente de logits exige uma camada softmax.")

        gradiente = np.array(gradiente, copy=True)
        if self.ativacao == "relu":
            # zera gradiente onde self.output <= 0
            # onde self.output > 0, gradiente passa sem alteração
            gradiente[self.output <= 0] = 0

        elif self.ativacao == "softmax" and not gradiente_logits:
            # produto soma gradiente * self.output por amostra
            produto = np.sum(gradiente * self.output, axis=1, keepdims=True)
            # gradiente = self.output * (gradiente - produto)
            gradiente = self.output * (gradiente - produto)

        # self.gradiente_pesos = self.inputs.T @ gradiente
        self.gradiente_pesos = self.inputs.T @ gradiente

        # self.gradiente_bias soma gradiente ao longo das amostras
        self.gradiente_bias = np.sum(gradiente, axis=0, keepdims=True)

        # devolve gradiente @ self.pesos.T pra camada anterior
        return gradiente @ self.pesos.T

    def atualizar_parametros(self, taxa_aprendizado):
        # subtrai taxa_aprendizado * self.gradiente_pesos de self.pesos
        self.pesos -= taxa_aprendizado * self.gradiente_pesos

        # subtrai taxa_aprendizado * self.gradiente_bias de self.bias
        self.bias -= taxa_aprendizado * self.gradiente_bias

    def relu(self, inputs):
        # retorna o maior valor entre 0 e cada elemento de inputs
        return np.maximum(0, inputs)

    def softmax(self, inputs):
        # exponenciais usa inputs menos o maior valor de cada amostra pra evitar overflow
        exponenciais = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # divide exponenciais pela soma de exponenciais de cada amostra
        return exponenciais / np.sum(exponenciais, axis=1, keepdims=True)
