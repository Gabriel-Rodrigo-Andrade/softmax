# MLP com backpropagation

Projeto da aula de redes neurais, baseado no notebook `Aula4MLP na mao.ipynb`.
A rede classifica imagens de dígitos de 0 a 9. O forward e o backpropagation
usam NumPy; o scikit-learn serve pra carregar os dados e separar treino e teste.

## Como rodar

Na raiz do projeto:

```sh
python3 -m pip install -r mlp_backpropagation/requirements.txt
python3 -m mlp_backpropagation.main
```

Com o ambiente virtual ativado, os mesmos comandos funcionam.

## Organização

- `layer.py`: camada densa, ReLU, softmax e gradientes dos parâmetros.
- `loss.py`: entropia cruzada e sua derivada combinada com softmax.
- `mlp.py`: liga as camadas, treina a rede e calcula a acurácia.
- `main.py`: carrega os dígitos, monta a rede e mostra os resultados.
- `__init__.py`: vazio. As importações são feitas direto de cada módulo.
- `requirements.txt`: bibliotecas necessárias.

## Rede e treinamento

A arquitetura ficou igual à do notebook: **64 → 5 → 15 → 10**.
São 64 pixels de entrada, duas camadas ocultas com ReLU e uma saída com softmax.
A classe prevista é a que tem a maior probabilidade.

Os pixels são divididos por 16 pra ficarem entre 0 e 1. A divisão dos dados
é de 80% pra treino e 20% pra teste, mantendo a proporção das classes.
O treinamento usa semente 0, 200 épocas, lotes de 32 e taxa de aprendizado 0,1.

Em cada lote, a rede faz o forward e calcula os gradientes de trás pra frente.
Só depois atualiza os pesos e bias. A perda e a acurácia do treino são
registradas no fim de cada época. O teste é avaliado depois do treinamento.

## Resultado registrado

| Medida | Resultado |
| --- | --- |
| Perda inicial de treino | 2,3029 |
| Perda final de treino | 0,0489 |
| Acurácia de treino | 97,98% |
| Acurácia de teste | 92,22% |

Esses valores são da execução com a configuração acima. A perda oscila em
algumas épocas por causa dos ajustes em mini-lotes. O programa treina do
zero a cada execução e não salva os pesos.
