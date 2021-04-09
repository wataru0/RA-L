# ラベルと論文のモデルの対応
Curriculum2-v1 -> CDR-v1
Curriculum-v4-16million -> CDR-v2

# 各手法のgeneralizationの可視化手順
- bash eval_generalization.sh　で，各手法のnpyデータを出力
- python plot_generalization.py でプロット

# 報酬関数の各項の可視化手順
- bash eval_generalization.sh　で，各手法のnpyデータを出力
- bash plot_TermOfRewardFunction.sh　で各手法の報酬関数の各項の値変化をプロット