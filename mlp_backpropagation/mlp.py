import numpy as np

from loss import CategoricalCrossEntropy
from layer import Layer


"""Rede MLP e treinamento em mini-lotes."""
class MLP:

    def __init__(self, camadas: list[Layer]):
        self.camadas = list(camadas)

        if not self.camadas:
            raise ValueError("A rede deve ter pelo menos uma camada.")

        self.criterio = CategoricalCrossEntropy()

    def prever_probabilidades(self, inputs):
        output = inputs

        # output de uma camada vira entrada da próxima camada.calcular_saida(output)
        for camada in self.camadas:
            output = camada.calcular_saida(output)

        return output

    def retropropagar_erro(self, output, targets):
        """Coordena o backpropagation de toda a rede; a atualização fica em treinar."""
        if self.camadas[-1].ativacao != "softmax":
            raise ValueError("O treinamento exige softmax na última camada.")

        # backward_softmax já deriva softmax + perda; gradiente_logits=True evita repetir essa derivada
        gradiente = self.criterio.backward_softmax(output, targets)
        gradiente = self.camadas[-1].calcular_gradientes(gradiente, gradiente_logits=True)

        # percorre as ocultas de trás pra frente; cada camada devolve gradiente pra anterior
        for camada in reversed(self.camadas[:-1]):
            gradiente = camada.calcular_gradientes(gradiente)

        return gradiente

    def treinar(
            self,
            inputs, 
            targets, 
            epocas=200, 
            taxa_aprendizado=0.1,
            tamanho_lote=32
    ):
        """Em cada lote: previsão, gradientes e atualização. Guarda as métricas por época."""
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
                output = self.prever_probabilidades(inputs[lote])
                self.retropropagar_erro(output, targets[lote])

                for camada in self.camadas:
                    camada.atualizar_parametros(taxa_aprendizado)

            output = self.prever_probabilidades(inputs)
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
