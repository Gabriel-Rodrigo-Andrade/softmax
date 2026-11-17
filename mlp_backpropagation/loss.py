import numpy as np


"""Perda de entropia cruzada para classificação multiclasse."""
class CategoricalCrossEntropy:

    def perdas_por_amostra(self, output, targets):
        targets = np.asarray(targets)
        # probabilidades limita output entre 1e-7 e 1 - 1e-7
        probabilidades = np.clip(output, 1e-7, 1 - 1e-7)

        if targets.ndim == 1:
            # confianca pega em probabilidades a classe indicada por targets
            confianca = probabilidades[np.arange(len(probabilidades)), targets]

        elif targets.ndim == 2:
            # confianca soma probabilidades * targets por amostra
            confianca = np.sum(probabilidades * targets, axis=1)

        else:
            raise ValueError("Targets devem ser índices de classes ou one-hot.")
        # retorna -np.log(confianca), a perda de cada amostra

        return -np.log(confianca)

    def loss(self, output, targets):
        # retorna a média de self.perdas_por_amostra(output, targets)
        return np.mean(self.perdas_por_amostra(output, targets))

    def backward_softmax(self, output, targets):
        """Derivada de softmax + entropia cruzada em relação aos logits.

        Recebe probabilidades e rótulos por índice ou one-hot. Os dois juntos
        evita dividir por probabilidades próximas de zero.
        """
        targets = np.asarray(targets)
        gradiente = np.array(output, copy=True)

        if targets.ndim == 1:
            # subtrai 1 de gradiente na classe indicada por targets
            gradiente[np.arange(len(output)), targets] -= 1

        elif targets.ndim == 2:
            # gradiente = gradiente - targets, com targets em one-hot
            gradiente -= targets

        else:
            raise ValueError("Targets devem ser índices de classes ou one-hot.")

        # divide gradiente por len(output) pra considerar a média do lote
        # clipping protege o log -> aqui usamos a derivada da perda original
        return gradiente / len(output)
