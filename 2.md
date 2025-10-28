# 1) Pulizia label & immagini

* **BBox fuori [0,1] o area=0** → correggi/clippa o scarta l’annotazione; se molte in una stessa immagine, rimuovi l’immagine dal train.
* **Immagini senza bbox** (ma dovrebbero averle) → spostale fuori dal train detection oppure etichettale (meglio rimuoverle subito dal train).
* **Duplicati / immagini corrotte** → elimina per non drogare le metriche.

# 2) Split robusto

* Se hai visto **sbilanciamento di classi** (rank/suit poco rappresentati) → fai split **stratificato** per presenza di classi/istanze e non solo random, così train/val hanno distribuzioni simili.

# 3) Scelte su input size, resize e letterbox

* **Aspect ratio molto variabile** → usa **letterbox** (stile YOLO) per mantenere il ratio e **multi-scale training** (es. 512–768 o 480–800) per robustezza.
* **Oggetti piccoli** (bbox mediane piccole) → **aumenta la risoluzione** di training (es. 640→768/896) o usa **mosaic** per far vedere più contesto mantenendo dimensioni relative degli oggetti.
* **Immagini già grandi ma oggetti grandi** → puoi restare più basso (es. 640) per velocità senza perdere troppo AP.

# 4) Bilanciamento classi (cards: rank/suit)

* **Classi minoritarie** →

  * **Oversampling** mirato (ripeti immagini con quelle classi);
  * **Augmentations class-aware** (applicate più spesso a campioni con classi rare);
  * In alternativa, **RepeatFactorSampler** / class weights durante il training.
* Nota: per i **semi rossi/neri**, non esagerare con **color jitter**: limita saturation/hue per non confondere seme/colore (p.es. hue ≤ ±0.02–0.05).

# 5) Anchors o anchor-free

* Dalla distribuzione **(w,h) delle bbox**:

  * Se usi un modello **con anchor** (YOLOv5/7/classici), calcola i **k-means dei box** (sull’immagine ridimensionata) per ottenere **anchor priors** migliori.
  * Se la variabilità è estrema o ci sono molti oggetti piccoli, valuta un **anchor-free** (YOLOX, RT-DETR, YOLOv8/11) per semplificare.

# 6) Policy di augmentation (derivata dai tuoi grafici)

* **Rotazioni e prospettiva**: per carte inclinate/occluse, abilita **random rotate** (±10–20°) e **perspective/affine** leggera.
* **Mosaic + MixUp**: ottimi per detection multi-oggetto (carte multiple).
* **Blur/Noise**: solo se nelle immagini reali c’è sfocatura/rumore.
* **Cutout**: utile per occlusioni parziali, ma non troppo aggressivo per non “tagliare” sistematicamente i semi.
* **Brightness/Contrast**: sì, ma moderato per non alterare eccessivamente i pattern dei semi.

# 7) Norm e pipeline coerente

* **Normalizzazione**: 0–1 (o mean/std del backbone, se richiesto).
* **Stessa pipeline** tra train e serving**: definisci ora le trasformazioni “inference-safe” (solo resize/letterbox + normalize), separate dalle augmentations di train.

# 8) Valutazione mirata

* Con classi sbilanciate, oltre alla **mAP@[.5:.95]**, guarda **AP per classe** (singoli rank/suit), **PR curve** e **confusion matrix** per capire quali semi/rank si confondono.
* Se hai task “detection + classification globale” (p.es. tipo di carta a livello immagine), mantieni metriche separate per i due task.

# 9) Miglioramenti dalla EDA specifica “carte”

* **Molti casi di rotazione**? → aumenta augment rotate e assicurati che il modello veda carte a 0°, 90°, 180°, 270°.
* **Tante carte per immagine**? → usa **NMS class-aware** e verifica la **recall** su crowd/overlap.
* **Sfondo uniforme** (tavolo) → aggiungi **background augmentation** (compositing su tavoli/texture diverse) per migliorare la generalizzazione.
* **Classi visivamente simili** (es. 6 vs 9, cuori vs quadri in piccolo) →

  * alza la risoluzione o
  * valuta **two-stage “detect → crop → classify”** (piccolo classifier sul crop per rank/suit, se la mAP detection è alta ma la classificazione fine è bassa).

# 10) Checklist pronta all’uso

* [ ] Fix/clip bbox + rimozione immagini problematiche
* [ ] Split stratificato per classi/istanze
* [ ] Scelta input size in base a bbox size (small → upsize)
* [ ] Calcolo **k-means anchors** (se anchor-based)
* [ ] Augmentations: mosaic, rotate, affine, brightness/contrast; jitter colore **limitato**
* [ ] Oversampling classi rare / sampler bilanciato
* [ ] Metriche per classe + PR curve + analisi errori
* [ ] YAML dataset + config modello coerenti con la pipeline

Se vuoi, posso:

* generarti un **YAML di dataset YOLO** (train/val/test) coerente con i tuoi path,
* proporre una **policy di augmentations** “starter” tarata su carte,
* calcolare qui gli **anchor priors** dai tuoi bbox (mi basta il CSV/JSON delle annotazioni).
