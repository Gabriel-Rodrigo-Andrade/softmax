# MLP com backpropagation

Projeto da aula de redes neurais, baseado no notebook `Aula4MLP na mao.ipynb`.
A rede classifica imagens de dígitos de 0 a 9. O forward e o backpropagation
usam NumPy; o scikit-learn serve pra carregar os dados e separar treino e teste.

## Como rodar

Na raiz do projeto:

```sh
python3 -m pip install -r mlp_backpropagation/requirements.txt
python3 mlp_backpropagation/main.py
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

## Onde está cada operação

Os nomes distinguem as operações da rede das contas de uma camada.
`camadas: list[Layer]` informa ao editor o tipo dos objetos para facilitar
a navegação até os métodos. `__init__` mantém o nome exigido pelo Python.

| Classe | Método e atalho | Nome anterior | Responsabilidade |
| --- | --- | --- | --- |
| `MLP` | [prever_probabilidades](/Users/gabrielrodrigoandrade/GRA/facul-IA/softmax/mlp_backpropagation/mlp.py:18) | `forward` | Passa os dados por todas as camadas e retorna as probabilidades da rede atual. |
| `Layer` | [calcular_saida](/Users/gabrielrodrigoandrade/GRA/facul-IA/softmax/mlp_backpropagation/layer.py:20) | `forward` | Calcula a saída de uma camada e aplica sua ativação. |
| `MLP` | [retropropagar_erro](/Users/gabrielrodrigoandrade/GRA/facul-IA/softmax/mlp_backpropagation/mlp.py:27) | `backward` | Coordena a volta pela rede, começando pela saída. |
| `Layer` | [calcular_gradientes](/Users/gabrielrodrigoandrade/GRA/facul-IA/softmax/mlp_backpropagation/layer.py:34) | `backward` | Calcula os gradientes de uma camada e devolve o gradiente das entradas. |
| `CategoricalCrossEntropy` | [backward_softmax](/Users/gabrielrodrigoandrade/GRA/facul-IA/softmax/mlp_backpropagation/loss.py:30) | `backward_softmax` | Calcula o gradiente inicial de softmax + entropia cruzada. |
| `Layer` | [atualizar_parametros](/Users/gabrielrodrigoandrade/GRA/facul-IA/softmax/mlp_backpropagation/layer.py:63) | `atualizar_parametros` | Usa os gradientes para atualizar pesos e bias. |
| `MLP` | [treinar](/Users/gabrielrodrigoandrade/GRA/facul-IA/softmax/mlp_backpropagation/mlp.py:42) | `treinar` | Organiza épocas e mini-lotes, chamando previsão, backward e atualização. |

O treinamento segue esta ordem:

```python
output = self.prever_probabilidades(inputs[lote])
self.retropropagar_erro(output, targets[lote])

for camada in self.camadas:
    camada.atualizar_parametros(taxa_aprendizado)
```

`prever_probabilidades` faz apenas o forward; não treina nem atualiza pesos.
`retropropagar_erro` começa em `backward_softmax` e chama `calcular_gradientes`
da saída e das camadas ocultas. Todos os gradientes são calculados antes
que `atualizar_parametros` altere os pesos e bias.

Na camada de saída, `gradiente_logits=True` significa que a derivada da
softmax já veio incluída em `backward_softmax`. Nas ocultas com ReLU,
`calcular_gradientes` aplica a derivada da ativação antes das contas dos pesos.
