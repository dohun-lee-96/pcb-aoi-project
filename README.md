# PCB-AOI Defect Detection

This project uses the Kaggle dataset `kubeedgeianvs/pcb-aoi` to detect PCB AOI defects with a lightweight YOLO object detection baseline.

## Project Goal

The model receives a PCB inspection image and predicts defect bounding boxes for two classes:

- `Bad_podu`
- `Bad_qiaojiao`

The project includes EDA, VOC XML to YOLO conversion, baseline model training, test evaluation, error analysis, and a Streamlit dashboard.

## Current Dataset Summary

- Raw images: 1,444
- Raw annotated objects: 5,764
- Original train images: 173
- Augmented train images: 1,211
- Test images: 60
- YOLO split after group-aware conversion: train 966, val 35, test 60
- Median bounding box area ratio: about 0.08% of image area

The very small object size is the main modeling challenge.

## Environment Notes

The current machine has an NVIDIA GPU visible through `nvidia-smi`, but the installed PyTorch package is CPU-only:

- Python: 3.12.10
- PyTorch: 2.12.0+cpu
- CUDA available in PyTorch: false

The baseline was therefore trained on CPU. If you later update the NVIDIA driver and install a CUDA-enabled PyTorch build compatible with your Python version, the scripts will automatically use GPU unless `--device cpu` is passed.

## Reproduce the Pipeline

Run commands from the `pcb_aoi_project` directory.

```bash
python -m pip install -r requirements.txt
```

Download the data using the token in the parent `.env.txt`:

```powershell
$env:KAGGLE_API_TOKEN=(Get-Content '..\.env.txt' | Select-String '^KAGGLE_API_TOKEN=').ToString().Substring('KAGGLE_API_TOKEN='.Length)
kaggle datasets download -d kubeedgeianvs/pcb-aoi -p data\raw --unzip
```

Generate EDA outputs:

```bash
python src/prepare_dataset.py
```

Convert XML annotations to YOLO format:

```bash
python src/convert_voc_to_yolo.py --val-ratio 0.2 --seed 42
```

Train the baseline:

```bash
python src/train_yolo.py --epochs 1 --imgsz 320 --batch 2
```

Evaluate on the test split:

```powershell
$weights=(python -c "import json; print(json.load(open('reports/metrics/training_summary.json'))['weights'])")
python src/evaluate.py --weights "$weights" --split test --imgsz 320 --conf 0.25
```

Create error analysis tables:

```bash
python src/error_analysis.py --split test --iou-threshold 0.5
```

Run lightweight operating-point improvement experiments:

```bash
python src/run_improvement_experiments.py
```

Run 5-fold Stratified Group CV operating-point experiments:

```bash
python src/run_cv_operating_point_experiments.py --epochs 1 --imgsz 320 --batch 2
```

Launch the dashboard:

```bash
streamlit run streamlit_app/app.py
```

## Baseline Results

The first baseline intentionally uses only 1 epoch on CPU to validate the full workflow.

- Test mAP50: 0.0522
- Test mAP50-95: 0.0106
- Mean precision: 0.3393
- Mean recall: 0.0644
- Matched detections at IoU 0.5: 36
- False positives: 19
- False negatives: 296
- Localization errors: 6

These results are low, which is expected for a 1 epoch CPU smoke-test baseline. The good news is that the project has clear room for improvement rather than being solved too easily.

## Completed Improvement Sweep

A lightweight inference-time sweep compared `imgsz` and confidence thresholds without retraining. The best review-oriented operating point was:

- Experiment: `op_img320_conf0p05`
- Image size: 320
- Confidence threshold: 0.05
- Matched detections: 168
- False positives: 326
- False negatives: 164
- Operating precision: 0.3401
- Operating recall: 0.5060

This setting reduces missed defects compared with the original `conf=0.25` baseline, but it also increases false positives. For factory inspection, this can be acceptable when the model is used as a candidate generator for human review.

## 5-Fold CV and Statistical Validation

A CPU-feasible 5-fold Stratified Group K-Fold experiment was run with `yolov8n`, `imgsz=320`, and 1 epoch per fold. The comparison focused on confidence operating points:

- `conf=0.25` had the best strict weighted score `false_negatives + 0.25 * false_positives`.
- `conf=0.10` is recommended as the review-oriented compromise because it lowers false negatives versus `conf=0.25` without the much larger false-positive load of `conf=0.05`.
- Friedman test on the weighted score: p-value `0.6752`.
- Wilcoxon signed-rank tests with Holm correction did not find statistically significant differences at alpha `0.05`.

Interpretation: the current 1-epoch CPU models are not strong enough for statistically proven model-quality differences. Treat `conf=0.10` as an operating-risk choice, then rerun the same framework with CUDA, 50+ epochs, higher image size, and model-capacity comparisons.

## Improvement Direction

- Train longer with a CUDA-enabled PyTorch build.
- Increase `imgsz` from 320 to 640 or 960 because defects are tiny.
- Tune confidence thresholds separately for inspection use cases where recall matters more than precision.
- Add class-aware augmentation or sampling because `Bad_qiaojiao` is much rarer.
- Compare `yolov8n`, `yolov8s`, and a high-resolution run.
- Review false negatives to separate true model misses from possible label quality issues.

## Important Outputs

- `reports/metrics/annotations.csv`: parsed annotation table
- `reports/metrics/eda_summary.json`: EDA summary
- `reports/figures/`: EDA plots
- `data/yolo/data.yaml`: YOLO dataset config
- `reports/metrics/evaluation_summary.json`: test metrics
- `reports/metrics/predictions.csv`: exported model predictions
- `reports/metrics/false_positives.csv`: false positive cases
- `reports/metrics/false_negatives.csv`: missed ground-truth boxes
- `reports/metrics/localization_errors.csv`: low-IoU prediction cases
