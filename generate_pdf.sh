#!/usr/bin/env bash
set -euo pipefail

# === CONFIG ===
ENV_PATH="./env/bin/activate"          # path al tuo venv
NOTEBOOK="1_Data_Exploration_PlayingCards.ipynb"
KERNEL_NAME="env"                      # nome kernel Jupyter legato al venv

dataset_paths=(
  "./datasets/Playing-Cards-Images-Object-Detection-Dataset/dataset_converted.csv"
  "./datasets/Playing-Cards-Object-Detection-Dataset/dataset_converted.csv"
  "./datasets/Playing-Cards-Labelized-Dataset/dataset_converted.csv"
)

# === ATTIVA VENV ===
if [[ -f "$ENV_PATH" ]]; then
  echo "📦 Attivo venv: $ENV_PATH"
  # shellcheck disable=SC1090
  source "$ENV_PATH"
else
  echo "❌ Venv non trovato in $ENV_PATH"
  exit 1
fi

# === CHECK DIPENDENZE ===
for cmd in jupyter python; do
  command -v "$cmd" >/dev/null || { echo "❌ '$cmd' mancante nel venv"; exit 1; }
done

# Assicura un kernelspec per questo venv
if ! jupyter kernelspec list 2>/dev/null | grep -qE "^\s*${KERNEL_NAME}\s"; then
  echo "🧩 Installo kernelspec Jupyter: ${KERNEL_NAME}"
  python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "Python (${KERNEL_NAME})"
fi

# === LOOP ===
for dataset_path in "${dataset_paths[@]}"; do
  if [[ ! -f "$dataset_path" ]]; then
    echo "⚠️  CSV non trovato, salto: $dataset_path"
    continue
  fi

  folder_name=$(basename "$(dirname "$dataset_path")")
  echo "➡️  Elaboro: $folder_name"

  executed_nb="output_${folder_name}.ipynb"
  pdf_name="${folder_name}.pdf"

  # 1) Esegui notebook usando il kernel del venv, passando DATASET_PATH al kernel
  DATASET_PATH="$dataset_path" \
  jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.kernel_name="${KERNEL_NAME}" \
    --output "$executed_nb" \
    "$NOTEBOOK"

  # 2) Converte l'eseguito in PDF (codice escluso)
  jupyter nbconvert --to pdf --TemplateExporter.exclude_input=True "$executed_nb"

  # 3) Rinomina PDF come richiesto
  mv "output_${folder_name}.pdf" "$pdf_name"

  echo "✅ Creato: $pdf_name"
done

echo "🎉 Fine."
