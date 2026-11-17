import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from layer import Layer
from loss import CategoricalCrossEntropy
from mlp import MLP


"""Treinamento da MLP com imagens de dígitos."""
def main():
    np.random.seed(0)
    digits = load_digits()

    # deixando os pixels entre 0 e 1
    inputs = digits.data / 16.0

    treino_inputs, teste_inputs, treino_targets, teste_targets = train_test_split(
        inputs, digits.target, test_size=0.2, random_state=0,
        stratify=digits.target,
    )

    rede = MLP([
        # oculta 1: recebe os 64 pixels e combina eles em 5 neurônios
        # a ReLU deixa passar os valores positivos e zera os negativos
        Layer(inputs.shape[1], 5, ativacao="relu"),

        # oculta 2: pega as 5 saídas anteriores e faz novas combinações
        # aqui são 15 neurônios, usando ReLU de novo
        Layer(5, 15, ativacao="relu"),

        # saída: recebe os 15 valores e gera uma pontuação pra cada dígito
        # a softmax transforma essas 10 pontuações em probabilidades de 0 a 9
        Layer(15, 10, ativacao="softmax"),
    ])

    criterio = CategoricalCrossEntropy()

    output = rede.prever_probabilidades(treino_inputs)
    print(f"Antes: perda={criterio.loss(output, treino_targets):.4f}, "
          f"acurácia de treino={rede.acuracia(output, treino_targets):.2%}")

    historico = rede.treinar(treino_inputs, treino_targets)

    for metricas in historico:
        if metricas["epoca"] == 1 or metricas["epoca"] % 20 == 0:
            print(f"Época {metricas['epoca']:3d}: "
                  f"perda={metricas['perda']:.4f}, "
                  f"acurácia de treino={metricas['acuracia']:.2%}")

    output = rede.prever_probabilidades(teste_inputs)

    print(f"Teste: perda={criterio.loss(output, teste_targets):.4f}, "
          f"acurácia={rede.acuracia(output, teste_targets):.2%}")
    print("Primeiros 10 rótulos reais:", teste_targets[:10])
    print("Primeiras 10 previsões:   ", np.argmax(output[:10], axis=1))


if __name__ == "__main__":
    main()
