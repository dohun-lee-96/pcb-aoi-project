from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "reports" / "metrics"
YOLO_DIR = ROOT / "data" / "yolo"


@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_label_boxes(image_id: str, split: str = "test") -> pd.DataFrame:
    data_yaml = load_json(METRICS_DIR / "class_mapping.json")
    inv_names = {v: k for k, v in data_yaml.items()}
    label_path = YOLO_DIR / "labels" / split / f"{image_id}.txt"
    image_path = next((YOLO_DIR / "images" / split).glob(f"{image_id}.*"), None)
    if not label_path.exists() or image_path is None:
        return pd.DataFrame()
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        cls, x, y, w, h = map(float, line.split())
        xmin = (x - w / 2) * width
        ymin = (y - h / 2) * height
        xmax = (x + w / 2) * width
        ymax = (y + h / 2) * height
        rows.append(
            {
                "image_id": image_id,
                "class_id": int(cls),
                "class_name": inv_names.get(int(cls), str(int(cls))),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )
    return pd.DataFrame(rows)


def draw_boxes(image_path: Path, gt: pd.DataFrame, pred: pd.DataFrame) -> object:
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for _, row in gt.iterrows():
        p1 = (int(row["xmin"]), int(row["ymin"]))
        p2 = (int(row["xmax"]), int(row["ymax"]))
        cv2.rectangle(image, p1, p2, (0, 180, 0), 2)
        cv2.putText(image, f"GT {row['class_name']}", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)

    for _, row in pred.iterrows():
        p1 = (int(row["xmin"]), int(row["ymin"]))
        p2 = (int(row["xmax"]), int(row["ymax"]))
        label = f"PR {row['class_name']} {row.get('confidence', 0):.2f}"
        cv2.rectangle(image, p1, p2, (220, 40, 40), 2)
        cv2.putText(image, label, (p1[0], max(15, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 40, 40), 1)
    return image


def recommendations_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority": "P0",
                "theme": "Train longer",
                "setting": "epochs 50-100, early stopping",
                "reason": "The current model is a 1-epoch CPU smoke test, so it has not learned enough object patterns.",
                "expected_effect": "Large recall and mAP gain",
            },
            {
                "priority": "P0",
                "theme": "Higher resolution",
                "setting": "imgsz 640, then 960 if memory allows",
                "reason": "Median box area is only about 0.08% of the image, so defects are very small.",
                "expected_effect": "Better small-defect recall",
            },
            {
                "priority": "P1",
                "theme": "Rare-class focus",
                "setting": "oversample images containing Bad_qiaojiao",
                "reason": "Bad_qiaojiao is less frequent and currently has near-zero detection performance.",
                "expected_effect": "Higher minority-class AP and recall",
            },
            {
                "priority": "P1",
                "theme": "Model capacity",
                "setting": "compare yolov8n, yolov8s, yolo11n",
                "reason": "Nano models are fast but may underfit subtle PCB texture defects.",
                "expected_effect": "Better localization and class separation",
            },
            {
                "priority": "P1",
                "theme": "Augmentation tuning",
                "setting": "mosaic on/off, scale range, hsv strength",
                "reason": "The dataset already includes augmented images; excessive augmentation may hurt fine defects.",
                "expected_effect": "More stable validation performance",
            },
            {
                "priority": "P2",
                "theme": "Threshold tuning",
                "setting": "confidence 0.05-0.50, NMS IoU 0.45-0.75",
                "reason": "Inspection tasks often prefer fewer missed defects even if false positives increase.",
                "expected_effect": "Controllable precision-recall tradeoff",
            },
        ]
    )


def experiment_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "Baseline",
                "experiment": "E00",
                "model": "yolov8n",
                "imgsz": 320,
                "epochs": 1,
                "sampling": "original split",
                "purpose": "Pipeline smoke test",
            },
            {
                "stage": "Baseline+",
                "experiment": "E01",
                "model": "yolov8n",
                "imgsz": 640,
                "epochs": 50,
                "sampling": "group split",
                "purpose": "Reasonable baseline for tiny objects",
            },
            {
                "stage": "Resolution",
                "experiment": "E02",
                "model": "yolov8n",
                "imgsz": 960,
                "epochs": 50,
                "sampling": "group split",
                "purpose": "Measure small-object resolution gain",
            },
            {
                "stage": "Capacity",
                "experiment": "E03",
                "model": "yolov8s",
                "imgsz": 640,
                "epochs": 50,
                "sampling": "group split",
                "purpose": "Check whether nano model underfits",
            },
            {
                "stage": "Imbalance",
                "experiment": "E04",
                "model": "yolov8s",
                "imgsz": 640,
                "epochs": 50,
                "sampling": "rare-class oversampling",
                "purpose": "Improve Bad_qiaojiao recall",
            },
            {
                "stage": "Augmentation",
                "experiment": "E05",
                "model": "yolov8s",
                "imgsz": 640,
                "epochs": 50,
                "sampling": "rare-class oversampling",
                "purpose": "Compare default vs reduced mosaic/hsv",
            },
            {
                "stage": "Operating point",
                "experiment": "E06",
                "model": "best previous",
                "imgsz": 640,
                "epochs": "-",
                "sampling": "-",
                "purpose": "Tune confidence and NMS thresholds",
            },
        ]
    )


def validation_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": "Split method",
                "design": "5-fold Stratified Group K-Fold",
                "why": "Keep augmented variants from the same original image in the same fold.",
            },
            {
                "item": "Primary metrics",
                "design": "mAP50, mAP50-95, recall, class-wise AP",
                "why": "Detection quality depends on both class prediction and localization.",
            },
            {
                "item": "OOF prediction",
                "design": "Save validation predictions from each fold",
                "why": "Use all training images once as validation for robust error analysis.",
            },
            {
                "item": "Model comparison",
                "design": "Wilcoxon signed-rank test on fold-level metrics",
                "why": "Compare paired fold results instead of relying on one lucky split.",
            },
            {
                "item": "Multiple experiments",
                "design": "Friedman test with Holm correction",
                "why": "Reduce false discoveries when many settings are compared.",
            },
            {
                "item": "Inspection risk",
                "design": "False negative rate and rare-class recall",
                "why": "A factory inspection model should avoid missed defects.",
            },
        ]
    )


def render_metric_gauge(label: str, value: float, target: float) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": label},
            gauge={"axis": {"range": [0, 1]}, "threshold": {"line": {"width": 3}, "value": target}},
            number={"valueformat": ".3f"},
        )
    )
    fig.update_layout(height=230, margin={"l": 20, "r": 20, "t": 40, "b": 20})
    st.plotly_chart(fig, use_container_width=True)


st.set_page_config(page_title="PCB-AOI Defect Detection", layout="wide")
st.title("PCB-AOI Defect Detection Dashboard")
st.caption("PCB-AOI object detection project: EDA, baseline evaluation, error review, and improvement roadmap.")

eda = load_json(METRICS_DIR / "eda_summary.json")
eval_summary = load_json(METRICS_DIR / "evaluation_summary.json")
error_summary = load_json(METRICS_DIR / "error_analysis_summary.json")
recommended_experiment = load_json(METRICS_DIR / "recommended_experiment.json")
cv_selected_candidate = load_json(METRICS_DIR / "cv_selected_candidate.json")
cv_friedman = load_json(METRICS_DIR / "cv_friedman.json")
cv_consensus = load_json(METRICS_DIR / "cv_consensus_recommendation.json")
predictions = load_csv(METRICS_DIR / "predictions.csv")
recommended_predictions = load_csv(METRICS_DIR / "predictions_recommended.csv")
improvement_experiments = load_csv(METRICS_DIR / "improvement_experiments.csv")
cv_results = load_csv(METRICS_DIR / "cv_operating_point_results.csv")
cv_summary = load_csv(METRICS_DIR / "cv_operating_point_summary.csv")
cv_wilcoxon = load_csv(METRICS_DIR / "cv_wilcoxon_holm.csv")
cv_folds = load_csv(METRICS_DIR / "cv_fold_summary.csv")
annotations = load_csv(METRICS_DIR / "annotations.csv")
false_positives = load_csv(METRICS_DIR / "false_positives.csv")
false_negatives = load_csv(METRICS_DIR / "false_negatives.csv")
localization_errors = load_csv(METRICS_DIR / "localization_errors.csv")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Images", eda.get("total_images", "-"))
col2.metric("Objects", eda.get("total_objects", "-"))
col3.metric("mAP50", f"{eval_summary.get('map50', 0):.3f}" if eval_summary else "-")
col4.metric("Recall", f"{eval_summary.get('recall_mean', 0):.3f}" if eval_summary else "-")

overview_tab, eda_tab, eval_tab, cv_tab, errors_tab, improve_tab = st.tabs(
    ["Overview", "EDA Insights", "Evaluation", "CV & Stats", "Error Review", "Improvement Plan"]
)

with overview_tab:
    st.subheader("Current Project Status")
    st.write(
        "The current model is a YOLO baseline trained for one epoch on CPU. "
        "It validates the pipeline, but it should not be treated as the final model."
    )
    if eda:
        split_rows = [
            {"split": split, "images": values["images"], "objects": values["objects"]}
            for split, values in eda.get("splits", {}).items()
        ]
        st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)
    if eval_summary:
        c1, c2 = st.columns(2)
        with c1:
            render_metric_gauge("mAP50", float(eval_summary.get("map50", 0)), 0.5)
        with c2:
            render_metric_gauge("Recall", float(eval_summary.get("recall_mean", 0)), 0.7)
    st.info(
        "Main diagnosis: defects are very small, class imbalance exists, and the baseline has many false negatives. "
        "The next experiments should prioritize recall and small-object detection."
    )
    if recommended_experiment:
        st.subheader("Best Operating Point Found So Far")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recommended imgsz", int(recommended_experiment.get("imgsz", 0)))
        c2.metric("Recommended conf", f"{recommended_experiment.get('conf', 0):.2f}")
        c3.metric("Operating recall", f"{recommended_experiment.get('operating_recall', 0):.3f}")
        c4.metric("False negatives", int(recommended_experiment.get("false_negatives", 0)))
        st.write(
            "Lowering the confidence threshold increased recall substantially, which is useful for inspection workflows "
            "where missed defects are more expensive than extra review candidates."
        )

with eda_tab:
    st.subheader("EDA-Based Modeling Decisions")
    if not annotations.empty:
        class_counts = annotations["class_name"].value_counts().reset_index()
        class_counts.columns = ["class_name", "objects"]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.bar(class_counts, x="class_name", y="objects", title="Object Count by Class"),
                use_container_width=True,
            )
        with c2:
            split_counts = annotations.groupby(["split", "class_name"]).size().reset_index(name="objects")
            st.plotly_chart(
                px.bar(split_counts, x="split", y="objects", color="class_name", title="Objects by Split"),
                use_container_width=True,
            )
        st.plotly_chart(
            px.histogram(annotations, x="box_area_ratio", color="class_name", nbins=40, title="Box Area Ratio"),
            use_container_width=True,
        )
        st.markdown(
            """
            **EDA decisions**

            - Use object detection rather than tabular ML because the target is defect location.
            - Use group-aware validation because augmented variants share the same original image.
            - Try larger `imgsz` values because bounding boxes are extremely small.
            - Track class-wise AP and recall because the minority class can be hidden by aggregate mAP.
            - Treat missing XML, broken images, invalid boxes, and coordinate overflow as data quality issues.
            """
        )
    else:
        st.info("Run `python src/prepare_dataset.py` first.")

with eval_tab:
    st.subheader("Model Evaluation")
    if eval_summary:
        metric_rows = pd.DataFrame(
            [
                {"metric": "mAP50", "value": eval_summary.get("map50", 0)},
                {"metric": "mAP50-95", "value": eval_summary.get("map50_95", 0)},
                {"metric": "Precision mean", "value": eval_summary.get("precision_mean", 0)},
                {"metric": "Recall mean", "value": eval_summary.get("recall_mean", 0)},
            ]
        )
        st.plotly_chart(px.bar(metric_rows, x="metric", y="value", range_y=[0, 1], title="Baseline Metrics"), use_container_width=True)
        st.dataframe(metric_rows, use_container_width=True, hide_index=True)
        st.json(eval_summary)
    else:
        st.info("Run training and `python src/evaluate.py` to populate evaluation metrics.")

    st.subheader("Recommended Experiment Matrix")
    st.dataframe(experiment_matrix(), use_container_width=True, hide_index=True)

    st.subheader("Completed Improvement Sweep")
    if not improvement_experiments.empty:
        display_cols = [
            "experiment_id",
            "imgsz",
            "conf",
            "predicted_objects",
            "matches",
            "false_positives",
            "false_negatives",
            "operating_precision",
            "operating_recall",
            "fn_fp_score",
        ]
        st.dataframe(improvement_experiments[display_cols], use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.scatter(
                improvement_experiments,
                x="false_positives",
                y="false_negatives",
                color="conf",
                symbol="imgsz",
                hover_name="experiment_id",
                title="False Positive vs False Negative Tradeoff",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(
                improvement_experiments.sort_values(["imgsz", "conf"]),
                x="conf",
                y="operating_recall",
                color="imgsz",
                markers=True,
                title="Recall by Confidence Threshold",
            ),
            use_container_width=True,
        )
    else:
        st.info("Run `python src/run_improvement_experiments.py` to populate improvement results.")

with cv_tab:
    st.subheader("5-Fold Stratified Group CV Results")
    st.caption(
        "CPU-feasible run: yolov8n, imgsz=320, 1 epoch per fold. "
        "The experiment compares confidence operating points on OOF validation predictions."
    )
    if not cv_summary.empty:
        st.dataframe(cv_summary, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.bar(
                    cv_summary,
                    x="candidate_id",
                    y="false_negatives_mean",
                    color="conf",
                    title="Mean False Negatives by Candidate",
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                px.bar(
                    cv_summary,
                    x="candidate_id",
                    y="false_positives_mean",
                    color="conf",
                    title="Mean False Positives by Candidate",
                ),
                use_container_width=True,
            )

        st.plotly_chart(
            px.scatter(
                cv_summary,
                x="false_positives_mean",
                y="false_negatives_mean",
                size="operating_recall_mean",
                color="conf",
                hover_name="candidate_id",
                title="OOF Compromise: False Positives vs False Negatives",
            ),
            use_container_width=True,
        )

        if cv_consensus:
            recommended = cv_consensus.get("recommended_consensus", {})
            strict = cv_consensus.get("strict_weighted_best", {})
            st.subheader("Selected Operating Point")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Consensus candidate", recommended.get("candidate_id", "-"))
            s2.metric("Confidence", f"{recommended.get('conf', 0):.2f}" if recommended else "-")
            s3.metric("OOF recall", f"{recommended.get('operating_recall_mean', 0):.3f}" if recommended else "-")
            s4.metric("Mean FN", f"{recommended.get('false_negatives_mean', 0):.1f}" if recommended else "-")
            st.info(cv_consensus.get("recommendation_reason", ""))
            st.write(
                f"Strict weighted-score best is `{strict.get('candidate_id', '-')}`. "
                "For inspection review, the dashboard recommends the consensus point because it reduces missed defects "
                "without the much larger false-positive load of the lowest threshold."
            )

        if not cv_results.empty:
            st.subheader("Fold-Level Metrics")
            st.plotly_chart(
                px.box(
                    cv_results,
                    x="candidate_id",
                    y="fn_fp_score",
                    points="all",
                    title="Fold-Level Weighted Error Score",
                ),
                use_container_width=True,
            )
            st.dataframe(cv_results, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python src/run_cv_operating_point_experiments.py` to populate 5-fold CV results.")

    st.subheader("Statistical Tests")
    if cv_friedman:
        st.json(cv_friedman)
    if not cv_wilcoxon.empty:
        st.dataframe(cv_wilcoxon, use_container_width=True, hide_index=True)
    else:
        st.info("Wilcoxon/Holm results are not available yet.")

    if not cv_folds.empty:
        st.subheader("Fold Construction")
        st.dataframe(cv_folds, use_container_width=True, hide_index=True)

with errors_tab:
    st.subheader("Error Analysis")
    if error_summary:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("GT objects", error_summary.get("ground_truth_objects", "-"))
        e2.metric("Predicted objects", error_summary.get("predicted_objects", "-"))
        e3.metric("False positives", error_summary.get("false_positives", "-"))
        e4.metric("False negatives", error_summary.get("false_negatives", "-"))
        st.json(error_summary)
    else:
        st.info("Run `python src/error_analysis.py` after prediction export.")

    review_type = st.radio(
        "Case table",
        ["Predictions", "False positives", "False negatives", "Localization errors"],
        horizontal=True,
    )
    tables = {
        "Predictions": predictions,
        "False positives": false_positives,
        "False negatives": false_negatives,
        "Localization errors": localization_errors,
    }
    selected_table = tables[review_type]
    if not selected_table.empty:
        st.dataframe(selected_table.head(200), use_container_width=True)
    else:
        st.info(f"No rows available for {review_type}.")

    st.subheader("Image Review")
    st.caption("Green boxes are ground truth. Red boxes are YOLO predictions.")
    prediction_source = st.radio(
        "Prediction source",
        ["Baseline predictions", "Recommended operating point"],
        horizontal=True,
    )
    active_predictions = (
        recommended_predictions
        if prediction_source == "Recommended operating point" and not recommended_predictions.empty
        else predictions
    )
    image_ids = sorted({p.stem for p in (YOLO_DIR / "images" / "test").glob("*.*")})
    if image_ids:
        selected = st.selectbox("Test image", image_ids)
        image_path = next((YOLO_DIR / "images" / "test").glob(f"{selected}.*"))
        gt = read_label_boxes(selected, "test")
        pred = active_predictions[active_predictions["image_id"] == selected] if not active_predictions.empty else pd.DataFrame()
        st.image(draw_boxes(image_path, gt, pred), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.write("Ground truth")
        c1.dataframe(gt, use_container_width=True)
        c2.write("Predictions")
        c2.dataframe(pred, use_container_width=True)
    else:
        st.info("YOLO test images are not available yet.")

with improve_tab:
    st.subheader("Detection Improvement Recommendations")
    st.dataframe(recommendations_table(), use_container_width=True, hide_index=True)

    if recommended_experiment:
        st.subheader("Recommendation Updated From Completed Sweep")
        st.success(
            f"Use `{recommended_experiment['experiment_id']}` for review-oriented inference: "
            f"`imgsz={int(recommended_experiment['imgsz'])}`, "
            f"`conf={recommended_experiment['conf']:.2f}`. "
            f"This reduced false negatives to {int(recommended_experiment['false_negatives'])} "
            f"with operating recall {recommended_experiment['operating_recall']:.3f}."
        )
        st.warning(
            "This is an operating-threshold improvement, not a final trained-model improvement. "
            "It trades more false positives for fewer missed defects."
        )

    st.subheader("Statistical Validation Plan")
    st.dataframe(validation_plan(), use_container_width=True, hide_index=True)
    st.markdown(
        """
        **How to decide whether a model is truly better**

        1. Run each candidate setting with 5-fold Stratified Group K-Fold.
        2. Save fold-level `mAP50`, `mAP50-95`, recall, and class-wise AP.
        3. Compare each improved setting against the baseline with a paired Wilcoxon signed-rank test.
        4. When comparing many settings, apply Friedman testing and Holm correction.
        5. Select the final model using validation OOF results, then evaluate once on the held-out test set.
        """
    )

    st.subheader("Next Best Experiments")
    if cv_consensus:
        st.success(cv_consensus.get("next_full_scale", "Run full-scale CV with CUDA and longer training."))
    else:
        st.success(
            "Best next step: install CUDA-enabled PyTorch if possible, then train `yolov8n` at `imgsz=640` for 50 epochs "
            "with group-aware validation. This directly targets the current false-negative problem."
        )
    st.warning(
        "Do not use augmented variants from the same original image across train and validation folds. "
        "That would leak visual information and overstate model quality."
    )
    st.markdown(
        """
        **Additional improvement ideas**

        - Compare `yolov8n` vs `yolov8s` with the same 5-fold split.
        - Increase `imgsz` to 640 because the defect boxes are tiny.
        - Add rare-class oversampling for images containing `Bad_qiaojiao`.
        - Train at least 50 epochs with early stopping; the current CV run is intentionally a CPU-feasible 1-epoch experiment.
        - Evaluate a review mode (`conf=0.10`) and a precision mode (`conf=0.25`) separately instead of forcing one threshold for every use case.
        """
    )
