import numpy as np

from .loss import CategoricalCrossEntropy


"""Rede MLP e treinamento em mini-lotes."""
class MLP:

    def __init__(self, camadas):
        self.camadas = list(camadas)

        if not self.camadas:
            raise ValueError("A rede deve ter pelo menos uma camada.")

        self.criterio = CategoricalCrossEntropy()

    def forward(self, inputs):
        output = inputs

        # output de uma camada vira entrada da próxima camada.forward(output)
        for camada in self.camadas:
            output = camada.forward(output)

        return output

    def backward(self, output, targets):
        """Percorre a rede de trás para frente sem atualizar os pesos."""
        if self.camadas[-1].ativacao != "softmax":
            raise ValueError("O treinamento exige softmax na última camada.")

        # gradiente vem de self.criterio.backward_softmax(output, targets)
        gradiente = self.criterio.backward_softmax(output, targets)
        gradiente = self.camadas[-1].backward(gradiente, gradiente_logits=True)

        # devolve gradiente pra camada anterior
        for camada in reversed(self.camadas[:-1]):
            gradiente = camada.backward(gradiente)

        return gradiente

    def treinar(
            self,
            inputs, 
            targets, 
            epocas=200, 
            taxa_aprendizado=0.1,
            tamanho_lote=32
    ):
        """Treina a rede e guarda a perda e a acurácia de cada época."""
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)

        if len(inputs) == 0 or len(inputs) != len(targets):
            raise ValueError("Inputs e targets devem ter o mesmo tamanho não vazio.")

        if epocas <= 0 or taxa_aprendizado <= 0 or tamanho_lote <= 0:
            raise ValueError("Épocas, taxa de aprendizado e lote devem ser positivos.")

        historico = []
        for epoca in range(1, epocas + 1):
            indices = np.random.permutation(len(inputs))

            for inicio in range(0, len(inputs), tamanho_lote):
                lote = indices[inicio:inicio + tamanho_lote]
                output = self.forward(inputs[lote])
                self.backward(output, targets[lote])

                for camada in self.camadas:
                    camada.atualizar_parametros(taxa_aprendizado)

            output = self.forward(inputs)
            historico.append({
                "epoca": epoca,
                "perda": self.criterio.loss(output, targets),
                "acuracia": self.acuracia(output, targets),
            })

        return historico

    def acuracia(self, output, targets):
        targets = np.asarray(targets)

        if targets.ndim == 2:
            # convertendo one-hot em índices
            targets = np.argmax(targets, axis=1)

        # a classe mais provável
        predicoes = np.argmax(output, axis=1)

        # calcula a proporção de acertos
        return np.mean(predicoes == targets)
