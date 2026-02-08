# AI Audio Enhancer

App em Streamlit para melhorar audios usando DeepFilterNet e uma equalizacao estilo
podcast (limpeza de graves, reforco de baixo e brilho).

## Recursos
- Reducao de ruido e realce de voz com DeepFilterNet.
- Conversao automatica para WAV mono via ffmpeg.
- Preview do audio e download do resultado final.

## Requisitos
- Python 3.11 (ver `runtime.txt`)
- ffmpeg disponivel no sistema (ver `packages.txt`)

## Instalacao
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Execucao
```bash
streamlit run main.py
```

## Como usar
1) Envie um arquivo de audio (wav, mp3, m4a, mp4).
2) Clique em "Melhorar audio".
3) Ouva o resultado e baixe o arquivo masterizado.

## Notas
- O ffmpeg e utilizado para converter para WAV mono com a taxa de amostragem do modelo.
- O processamento pode levar alguns segundos dependendo do tamanho do audio e do hardware.
