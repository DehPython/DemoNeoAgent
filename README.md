# Como rodar

Na pasta principal DemoNeoAgent

```bash
uvicorn backend.main:app --reload
```

Caso dê erro de porta já estar sendo utilizada, tente assim

```bash
uvicorn backend.main:app --reload --port 8080
```

Você pode escolher o valor da porta.

## Habilitando a opção de usar VOZ (ASR)

Para que a funcionalidade de gravação de voz (ASR) funcione corretamente, você deve seguir a ordem abaixo rigorosamente:

### Passo 1: Instale o FFmpeg no seu computador (Obrigatório)
O FFmpeg precisa ser instalado de fato na sua máquina (Sistema Operacional), não é apenas um pacote Python.
* **Windows:** Procure o instalador no [site oficial](https://www.ffmpeg.org/download.html) ou, pelo terminal/powershell, use `winget install ffmpeg`
* **Linux (Debian/Ubuntu):** Rode no terminal `sudo apt update && sudo apt install ffmpeg`
* **Mac:** Rode no terminal `brew install ffmpeg`

### Passo 2: Verifique a sua versão do CUDA (Se tiver GPU NVIDIA)
Caso você tenha uma placa de vídeo da NVIDIA, rodaremos nela para maior performance. Verifique a versão do seu `CUDA` suportada rodando no terminal:
```bash
nvidia-smi
```
*(Olhe a "CUDA Version" no canto superior direito).*
Se você **não** tiver uma placa de vídeo dedicada, você deve usar a instalação para **CPU**.

### Passo 3: Escolha e instale a versão correta do PyTorch
Só avance para este passo após achar sua versão no Passo 2. Copie e rode **somente um** dos comandos abaixo (escolha o que bate com seu sistema):

* **Apenas CPU (Sem placa de vídeo):**
  ```bash
  uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
  ```
* **Placa de vídeo com CUDA 12.6:**
  ```bash
  uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
  ```
* **Placa de vídeo com CUDA 12.8:**
  ```bash
  uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
  ```
* **Placa de vídeo com CUDA 13.0:**
  ```bash
  uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
  ```

### Passo 4: Por fim, instale as dependências do ASR
Somente após ter concluído todos os passos anteriores (FFmpeg instalado e PyTorch correto para sua máquina), rode o comando final:

```bash
uv pip install -r asr_requirements.txt
```

Prontinho, agora é só rodar o servidor normalmente e a opção de falar estará habilitada!
