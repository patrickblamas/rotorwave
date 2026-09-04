@echo off
REM Instala o pacote rotorwave no Python que estiver ativo.
REM Use com duplo clique a partir do Anaconda Prompt, ou clique duas vezes no
REM Explorador de Arquivos se o Anaconda estiver no PATH do sistema.

cd /d "%~dp0"
echo.
echo === Pasta do projeto: %CD%
echo.

if not exist "pyproject.toml" (
    echo ERRO: nao encontrei o arquivo pyproject.toml nesta pasta.
    echo Descompacte o zip de verdade ^(botao direito ^> Extrair tudo^) e rode
    echo este arquivo de dentro da pasta que contem o pyproject.toml.
    echo.
    pause
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo ERRO: nenhum "python" encontrado no PATH.
    echo Abra o Anaconda Prompt ^(menu Iniciar^) e rode este arquivo de la:
    echo     cd /d "%CD%"
    echo     instalar_windows.bat
    echo.
    pause
    exit /b 1
)

echo === Python usado:
python -c "import sys; print(sys.version); print(sys.executable)"
echo.

echo === Instalando (pip install -e .) ...
python -m pip install -e .
if errorlevel 1 (
    echo.
    echo A instalacao falhou. Veja a mensagem acima e consulte INSTALACAO.md.
    pause
    exit /b 1
)

echo.
echo === Testando a instalacao ...
python -c "from rotorwave import compressor_rotor; print(compressor_rotor(11).summary())"
if errorlevel 1 (
    echo.
    echo O pacote foi instalado mas o teste falhou. Consulte INSTALACAO.md.
    pause
    exit /b 1
)

echo.
echo === Tudo certo! Agora voce pode rodar, por exemplo:
echo     python examples\fig04_dispersion_omega_k.py
echo.
pause
