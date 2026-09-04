# Como rodar no Windows com Anaconda

Três caminhos, do mais simples ao mais completo. Qualquer um funciona — escolha um.

---

## Caminho 1 — Rodar sem instalar nada (mais rápido)

1. Descompacte o `rotorwave.zip` numa pasta qualquer, por exemplo
   `C:\Users\<seu usuário>\Desktop\rotorwave`.
   **Importante:** o Windows deixa você "abrir" o zip com um duplo clique sem
   descompactar de verdade — nesse modo nada funciona. Clique com o botão direito no
   arquivo → **Extrair tudo…**
2. Abra o **Spyder** (pelo Anaconda Navigator ou pelo menu Iniciar).
3. `File → Open…` e escolha o arquivo **`run_me.py`** que está na raiz da pasta.
4. Aperte **F5**.

O `run_me.py` ajusta o caminho do Python sozinho, então não precisa instalar nada. Ele
roda a análise do rotor de 11 discos e abre os gráficos. Os arquivos dentro de
`examples/` também funcionam assim: abra e aperte F5.

Se os gráficos não aparecerem no Spyder, vá em
`Ferramentas → Preferências → IPython console → Graphics → Graphics backend` e escolha
`Inline` (ou `Automatic`, se quiser janelas separadas).

---

## Caminho 2 — Instalar o pacote (recomendado para usar em outros scripts)

Instalando, você pode escrever `from rotorwave import ...` em qualquer script ou
notebook, de qualquer pasta.

1. Abra o **Anaconda Prompt** (menu Iniciar → digite "Anaconda Prompt").
2. Vá até a pasta onde você descompactou. Exemplo:

   ```bat
   cd %USERPROFILE%\Desktop\rotorwave
   ```

   Se a pasta estiver em outro disco (por exemplo `D:`), rode antes `D:` para trocar de
   disco.

3. Confira que você está na pasta certa — o comando abaixo tem que listar `pyproject.toml`:

   ```bat
   dir
   ```

4. Instale:

   ```bat
   pip install -e .
   ```

   O ponto final faz parte do comando. O `-e` instala em modo "editável": se você mexer
   no código, a mudança vale na hora, sem reinstalar.

5. Teste:

   ```bat
   python -c "from rotorwave import reference_rotor; print(reference_rotor(11).summary())"
   ```

   Deve imprimir a descrição do rotor. A partir daqui, `python examples\dispersion_omega_k.py`
   e os demais exemplos rodam de qualquer lugar.

**Atalho:** dê um duplo clique em `instalar_windows.bat` — ele faz os passos 2 a 5
sozinho e deixa a janela aberta para você ler o resultado.

---

## Caminho 3 — Ambiente conda separado (mais limpo, não mexe no seu `base`)

Recomendado se você não quer arriscar mudar as versões de numpy/scipy do seu Anaconda.

```bat
conda create -n rotorwave python=3.11 numpy scipy matplotlib pytest -y
conda activate rotorwave
cd %USERPROFILE%\Desktop\rotorwave
pip install -e .
python examples\dispersion_omega_k.py
```

Para usar esse ambiente no Spyder, instale o Spyder dentro dele
(`conda install -n rotorwave spyder -y`) ou aponte o interpretador em
`Ferramentas → Preferências → Python interpreter`.

---

## Usar no Jupyter

Abra o notebook `rotorwave_demo.ipynb` (Anaconda Navigator → Jupyter Notebook → navegue
até a pasta). A primeira célula resolve o caminho automaticamente, mesmo sem instalação.

---

## Rodar os testes (opcional, confirma que está tudo certo)

```bat
cd %USERPROFILE%\Desktop\rotorwave
pytest
```

Devem passar 30 testes em cerca de 20 segundos.

---

## Problemas comuns

**`ModuleNotFoundError: No module named 'rotorwave'`**
O pacote não foi instalado e você rodou um script que não é o `run_me.py` nem está em
`examples/`. Use o Caminho 2, ou copie estas linhas para o topo do seu script:

```python
import sys
sys.path.insert(0, r"C:\Users\<seu usuário>\Desktop\rotorwave\src")
```

**`'pip' não é reconhecido como um comando interno ou externo`**
Você abriu o Prompt de Comando comum em vez do **Anaconda Prompt**. Abra o Anaconda
Prompt; ele já vem com o Python e o pip do Anaconda configurados.

**`ERROR: file:///... does not appear to be a Python project`**
Você não está na pasta certa. Rode `dir` e confirme que aparece `pyproject.toml`. Se
aparecer só uma subpasta `rotorwave`, entre nela com `cd rotorwave`. Isso acontece quando
o zip é extraído criando uma pasta dentro da outra.

**`ModuleNotFoundError: No module named 'matplotlib'`**
Rode `conda install matplotlib -y`. O Anaconda padrão já traz numpy, scipy e matplotlib,
então isso só aparece em ambientes novos e enxutos.

**Acesso negado / permissão ao instalar**
Use o Caminho 3 (ambiente separado) ou rode o Anaconda Prompt como administrador.

**Os gráficos não abrem**
No Spyder, verifique o backend gráfico (fim do Caminho 1). Rodando pelo Anaconda Prompt,
os scripts salvam PNGs em `examples\figures\` de qualquer forma — pode abrir os arquivos
direto de lá.

---

## Versões

Precisa de Python 3.10 ou mais novo, numpy ≥ 1.23, scipy ≥ 1.10 e matplotlib ≥ 3.6 para
os gráficos. Qualquer Anaconda dos últimos anos atende. Para conferir:

```bat
python --version
python -c "import numpy, scipy, matplotlib; print(numpy.__version__, scipy.__version__, matplotlib.__version__)"
```
