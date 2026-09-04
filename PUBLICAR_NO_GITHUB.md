# Publicando o rotorwave no GitHub

Guia passo a passo, pensado para quem nunca usou Git. Ao final você terá o repositório
público, um DOI para citar no artigo e os testes rodando automaticamente a cada
alteração.

---

## Passo 0 — Antes de publicar (5 minutos, mas importante)

- [ ] **Decidir a licença** (veja a seção "Licença" mais abaixo). O arquivo `LICENSE`
      já vem com a MIT; se quiser outra, troque antes do primeiro commit.
- [ ] **Confirmar a titularidade.** O código foi produzido no âmbito de um doutorado na
      USP com financiamento CNPq/FAPESP/CAPES. Vale confirmar com o Prof. Nicoletti e,
      se for o caso, com a Agência USP de Inovação, se há alguma exigência quanto a
      licenciamento de software. Na prática, código de apoio a artigo publicado em
      acesso aberto é rotineiramente liberado com licença permissiva, mas a
      confirmação é rápida e evita dor de cabeça depois.
- [ ] **Alinhar com o coautor.** Ele aparece como autor no `pyproject.toml`, no
      `CITATION.cff` e no `LICENSE` — combine antes de publicar.
- [x] ~~Trocar os marcadores `<user>`~~ — já feito: `pyproject.toml`, `README.md` e
      `CITATION.cff` apontam para `github.com/patrickblamas/rotorwave`.
- [ ] **Conferir o que NÃO vai subir.** O `.gitignore` já exclui `__pycache__/`,
      `examples/figures/`, `.pytest_cache/` e afins. As figuras são geradas pelos
      scripts, então não precisam ir para o repositório (se você quiser mostrá-las no
      README, é melhor colocar 2 ou 3 numa pasta `docs/img/` e referenciá-las).

---

## Passo 1 — Criar a conta e o repositório

1. Crie uma conta em <https://github.com> (se ainda não tiver). Vale usar o e-mail
   institucional e ativar o **GitHub Education** (grátis, dá acesso a recursos extras).
2. Clique em **New repository** (botão verde, ou <https://github.com/new>).
3. Preencha:
   - **Repository name:** `rotorwave`
   - **Description:** `Wave propagation analysis of rotors with longitudinal periodicity`
   - **Public** (precisa ser público para o artigo poder citá-lo e para o Zenodo)
   - **NÃO** marque "Add a README", "Add .gitignore" nem "Choose a license" — você já
     tem os três na pasta, e marcar isso cria conflito no primeiro envio.
4. **Create repository**. A página que aparece mostra a URL do repositório; deixe-a
   aberta.

---

## Passo 2 — Enviar os arquivos

Escolha **um** dos dois caminhos.

### Caminho A — GitHub Desktop (recomendado, sem terminal)

1. Baixe e instale o **GitHub Desktop**: <https://desktop.github.com>
2. Abra, faça login com sua conta.
3. `File → Add local repository…` → escolha `C:\Users\patri\Desktop\rotorwave`.
4. Ele vai avisar que a pasta não é um repositório Git ainda e oferecer
   **"create a repository"** — clique nisso. Na tela que abrir, deixe o nome
   `rotorwave` e **não** marque para adicionar licença ou .gitignore (já existem).
5. Na aba **Changes** aparecem todos os arquivos. Escreva em *Summary*:
   `Initial release: rotorwave 1.0.0` e clique em **Commit to main**.
6. Clique em **Publish repository** no topo. Desmarque *"Keep this code private"*.

Pronto. Daí em diante, toda vez que você editar um arquivo, ele aparece na aba
**Changes**: escreva um resumo, **Commit**, depois **Push origin**. É assim que suas
alterações nos exemplos (as que você já fez) ficam registradas e recuperáveis.

### Caminho B — Linha de comando (Anaconda Prompt)

Instale o Git para Windows (<https://git-scm.com/download/win>), depois:

```bat
cd %USERPROFILE%\Desktop\rotorwave

git init -b main
git config user.name "Patrick B. Lamas"
git config user.email "seu-email@usp.br"

git add .
git status                     REM confira: nao deve listar __pycache__ nem figures/
git commit -m "Initial release: rotorwave 1.0.0"

git remote add origin https://github.com/patrickblamas/rotorwave.git
git push -u origin main
```

Na primeira vez o Git abre uma janela pedindo login no GitHub — autorize pelo navegador.

Depois disso, o ciclo do dia a dia é sempre o mesmo:

```bat
git add -A
git commit -m "descricao curta do que mudou"
git push
```

---

## Passo 3 — Criar uma versão (release)

Um *release* congela um estado do código com um número de versão. É o que o Zenodo usa
para gerar o DOI.

1. Na página do repositório: **Releases** (coluna da direita) → **Create a new release**.
2. **Choose a tag** → digite `v1.0.0` → *Create new tag on publish*.
3. **Release title:** `rotorwave 1.0.0`
4. Descrição: algo como
   *"First public release, accompanying Lamas & Nicoletti (2024), JSV 571, 118095.
   Reproduces the dispersion diagrams, band gaps and wave-Campbell diagram of the paper."*
5. **Publish release**.

---

## Passo 4 — DOI pelo Zenodo (o passo que importa para o artigo)

Um link do GitHub pode mudar ou sumir; um DOI é permanente. Revistas gostam disso, e
algumas exigem.

1. Entre em <https://zenodo.org> e faça login **com a conta do GitHub**.
2. Vá em <https://zenodo.org/account/settings/github/> — ele lista seus repositórios.
3. Ligue a chavinha do `rotorwave`.
4. **Volte ao GitHub e crie o release** (Passo 3). O Zenodo só arquiva releases criados
   *depois* de ligar a chave — se você já criou o release antes, apague-o e crie de novo,
   ou publique um `v1.0.1`.
5. Em alguns minutos o Zenodo gera o DOI. Copie o **badge Markdown** que ele oferece e
   cole no topo do `README.md`, logo abaixo do título:

   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```

6. Atualize também o `CITATION.cff` com o DOI e a URL definitiva.

O Zenodo dá dois DOIs: um para **cada versão** e um **"concept DOI"** que aponta sempre
para a versão mais recente. No artigo, cite o DOI da versão que você usou para gerar os
resultados — assim quem for reproduzir pega exatamente o mesmo código.

---

## Passo 5 — O que escrever no artigo

Sugestão de *Code availability statement* (a maioria das revistas Elsevier pede isso
numa seção própria, antes das referências):

> **Code availability.** The Python implementation described in this work is openly
> available at https://github.com/patrickblamas/rotorwave under the MIT license, and archived
> at https://doi.org/10.5281/zenodo.XXXXXXX. The repository includes the scripts that
> reproduce every figure presented here.

E na lista de referências, o software entra como uma entrada própria (o `CITATION.cff`
do repositório gera o BibTeX automaticamente; no GitHub, em *"Cite this repository"*, na
coluna da direita).

---

## Passo 6 — Ajustes finais no repositório (opcional, mas rende)

- **About** (engrenagem ao lado da descrição, à direita): adicione a URL do DOI e os
  *topics*: `rotordynamics`, `wave-propagation`, `band-gap`, `periodic-structures`,
  `bloch-floquet`, `finite-element`, `python`. É assim que outras pessoas acham o
  repositório.
- **Badge dos testes.** O arquivo `.github/workflows/tests.yml` já está incluído: assim
  que você fizer o push, o GitHub roda os 30 testes em Python 3.10, 3.11 e 3.12 a cada
  alteração, de graça. Para exibir o resultado, cole no topo do README:

  ```markdown
  [![tests](https://github.com/patrickblamas/rotorwave/actions/workflows/tests.yml/badge.svg)](https://github.com/patrickblamas/rotorwave/actions/workflows/tests.yml)
  ```

- **Uma figura no README** ajuda muito quem chega pela primeira vez. Coloque, por
  exemplo, `directional_disp_dFRF.png` em `docs/img/` e referencie com
  `![](docs/img/directional_disp_dFRF.png)`.
- **PyPI** (`pip install rotorwave` para qualquer um): só se você quiser manter o pacote
  a longo prazo. Não é necessário para o artigo — o GitHub + Zenodo bastam.

---

## Licença: o que é e como escolher

Sem licença explícita, vale o padrão da lei de direitos autorais: **ninguém pode usar,
copiar ou modificar** o seu código legalmente, mesmo estando ele visível no GitHub. Ou
seja, publicar sem licença anula o propósito de disponibilizar o código junto ao artigo.
Por isso o repositório já vem com um arquivo `LICENSE`.

As quatro opções realistas para código científico:

| Licença | Em uma frase | Efeito prático |
|---|---|---|
| **MIT** (a que está lá) | "Use como quiser, só mantenha o aviso de copyright." | Máxima adoção. Empresas e outros grupos podem usar sem atrito. |
| **BSD 3-Clause** | Igual à MIT, mais uma cláusula que impede usar o nome dos autores para promover derivados. | Comum em pacotes científicos (numpy, scipy). Diferença mínima na prática. |
| **Apache 2.0** | Igual à MIT, mais concessão explícita de patentes e exigência de registrar alterações. | Mais formal; boa se houver qualquer discussão de patente envolvida. |
| **GPL 3.0** | "Use como quiser, mas qualquer derivado também precisa ser aberto." | Protege contra apropriação fechada, mas afasta uso industrial e impede combinar com pacotes de licença permissiva. |

Para um código que acompanha artigo, cujo objetivo é **ser usado e citado**, a MIT (ou
a BSD-3) é o padrão da área — é o que numpy, scipy e a maioria dos pacotes de dinâmica
de rotores usam. A GPL faria sentido se a preocupação principal fosse impedir que alguém
feche um derivado, o que raramente é o caso aqui.

Três esclarecimentos que costumam gerar confusão:

1. **A licença do código não tem relação com os direitos do artigo.** O artigo publicado
   na JSV é regido pelo contrato com a Elsevier; o código é obra separada e a titularidade
   é de quem o escreveu (e/ou da instituição). Publicar o código com licença MIT não
   conflita com o contrato do artigo.
2. **Licenciar não é abrir mão da autoria.** Você continua sendo o autor; a licença só
   diz o que os outros podem fazer. O aviso de copyright no `LICENSE` permanece, e a MIT
   exige que ele seja mantido em qualquer cópia.
3. **Preencha o titular corretamente.** Hoje o `LICENSE` diz
   `Copyright (c) 2026 P. B. Lamas and R. Nicoletti`. Se a USP exigir figurar como
   titular, o formato usual é
   `Copyright (c) 2026 Universidade de São Paulo`, ou os dois nomes. Ajuste antes do
   primeiro commit — depois de publicado, mudar licença exige o consentimento de todos
   que contribuíram.

Se quiser trocar a licença, o texto de qualquer uma delas está em
<https://choosealicense.com> — é só substituir o conteúdo do arquivo `LICENSE` e ajustar
a linha `license` no `pyproject.toml` e no `CITATION.cff`.

---

## Erros comuns na primeira vez

**"Updates were rejected because the remote contains work that you do not have"**
Você marcou "Add a README" ao criar o repositório. Resolva com
`git pull --rebase origin main` e depois `git push`, ou apague o repositório no GitHub e
crie de novo sem marcar nada.

**Subiu `__pycache__` ou as figuras sem querer**
Elas já estão no `.gitignore`, mas se algum arquivo foi commitado antes disso:
`git rm -r --cached src/rotorwave/__pycache__ examples/figures` e commit de novo.

**Commitou um arquivo grande ou sigiloso**
Enquanto for só o último commit e você ainda não deu push:
`git reset --soft HEAD~1`, remova o arquivo, commit de novo. Depois do push é bem mais
trabalhoso — vale conferir o `git status` antes.

**Quer desfazer alterações locais em um arquivo**
No GitHub Desktop: botão direito no arquivo na aba Changes → *Discard changes*.
Na linha de comando: `git checkout -- caminho/do/arquivo.py`.
