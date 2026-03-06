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

## Habilitando o ASR

### Verifique se você tem o ffmpeg instalado

FFMPEG é uma biblioteca padrão para manipulação de audio e de imagens.

[Site do FFMPEG](https://www.ffmpeg.org/download.html)

Se não tiver baixe para seu sistema.

### Verifique sua versão do CUDA

```bash
nvidia-smi
```

Abra o arquivo asr_requirements.txt, e baixe de acordo com sua versão CUDA.

Em seguida, instale as dependencias do ASR.

```bash
uv pip install -r asr_requirements.txt
```

Prontinho, agora é só rodar normalmente que a opção de falar estará rodando normalmente.
