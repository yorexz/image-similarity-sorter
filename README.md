## 🖼️ Image Similarity Sorter

Uma ferramenta de linha de comando baseada em machine learning que organiza imagens por similaridade visual, usando redes neurais convolucionais e algoritmos de redução dimensional.

## 📋 Sobre o Projeto

Este aplicativo Python analisa e ordena imagens com base em sua similaridade visual. Utilizando um modelo de deep learning pré-treinado (ResNet50), ele extrai características de cada imagem e as organiza em uma sequência que agrupa imagens visualmente semelhantes.

Diferentemente de organizadores que dependem de metadados (como data, tamanho ou nome do arquivo), esta ferramenta analisa o conteúdo visual das imagens, permitindo organizar fotos por cenas, cores dominantes, objetos ou pessoas, mesmo sem tags.

## ✨ Funcionalidades

- Ordenação por similaridade visual utilizando deep learning; 
- Processamento in-place que reorganiza os arquivos de forma segura no local original;
- Suporte a vários formatos de imagem (JPG, PNG, etc.);
- Visualização de progresso com barras de carregamento em tempo real;
- Logging detalhado para acompanhamento e diagnóstico.

## 🛠️ Tecnologias Utilizadas

- TensorFlow/Keras: Para processamento com redes neurais convolucionais;
- ResNet50: Modelo pré-treinado para extração de características visuais;
- t-SNE: Algoritmo para redução de dimensionalidade e agrupamento;
- scikit-learn: Para cálculos de similaridade e manipulação de dados;
- Pillow (PIL): Para processamento básico de imagens;
- tqdm: Para visualização de progresso em tempo real.

## 🚀 Instalação e Uso

### Pré-requisitos:

- Python 3.6 ou superior;
- Memória RAM suficiente para processar as imagens (recomendado mínimo 4GB).

### Instalação

Clone este repositório:

```bash
git clone https://github.com/yorexz/image-similarity-sorter.git
cd image-similarity-sorter
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

### Uso

Para ordenar todas as imagens em uma pasta:

```bash
python src/image_similarity.py /caminho/para/pasta/de/imagens
```
Após a execução, as imagens serão renomeadas para uma sequência numerada (0001.jpg, 0002.jpg, etc.) onde imagens adjacentes têm maior similaridade visual.

## 🖥️ Como Funciona?

- Extração de Características: O algoritmo carrega cada imagem e a processa através do modelo ResNet50 para extrair um vetor de características de 2048 dimensões que representa os aspectos visuais da imagem;
- Cálculo de Similaridade: É construída uma matriz de similaridade de cosseno entre todas as imagens para determinar quão semelhantes elas são;
- Mapeamento Dimensional: O algoritmo t-SNE reduz os vetores de alta dimensão para um espaço unidimensional, preservando a estrutura de similaridade;
- Reorganização Segura: As imagens são renomeadas em um processo de duas etapas para garantir que nenhum arquivo seja perdido durante a reorganização.

## 📊 Casos de Uso

Fotógrafos: Organizar sessões de fotos por composição visual ou tema;
Designers: Agrupar elementos de design semelhantes;
Pesquisadores: Organizar datasets de imagens para treinamento de ML;
Coleções Pessoais: Ordenar fotos por similaridade antes de criar álbuns.
