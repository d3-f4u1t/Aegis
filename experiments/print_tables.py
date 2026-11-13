import json

with open("experiments/results/phase5_summary.json", "r") as fh:
    data = json.load(fh)

for exp_key, exp_data in data.items():
    print(f"\n{'='*30} {exp_key.upper()} {'='*30}")
    print(f"Dataset size: {exp_data['dataset_size']} | Train: {exp_data['n_train']} | Val: {exp_data['n_val']} | Test: {exp_data['n_test']} | Positives in test: {exp_data['n_positive_test']}")
    print("-" * 115)
    print(f"{'Model':20s} | {'Acc':6s} | {'Prec':6s} | {'Rec':6s} | {'F1':6s} | {'FPR':6s} | {'FNR':6s} | {'PR-AUC':6s} | {'AP':6s} | {'Lat(us)':7s} | {'TP':4s} | {'FP':4s} | {'TN':4s} | {'FN':4s}")
    print("-" * 115)
    for model_name, m in exp_data["model_results"].items():
        acc = f"{m['accuracy']:.4f}"
        prec = f"{m['precision']:.4f}"
        rec = f"{m['recall']:.4f}"
        f1 = f"{m['f1']:.4f}"
        fpr = f"{m['false_positive_rate']:.4f}"
        fnr = f"{m['false_negative_rate']:.4f}"
        prauc = f"{m['pr_auc']:.4f}"
        ap = f"{m['average_precision']:.4f}"
        lat = f"{m['inference_latency_ms_per_sample']*1000.0:.2f}"
        tp = int(m['tp'])
        fp = int(m['fp'])
        tn = int(m['tn'])
        fn = int(m['fn'])
        print(f"{model_name:20s} | {acc:6s} | {prec:6s} | {rec:6s} | {f1:6s} | {fpr:6s} | {fnr:6s} | {prauc:6s} | {ap:6s} | {lat:7s} | {tp:4d} | {fp:4d} | {tn:4d} | {fn:4d}")
