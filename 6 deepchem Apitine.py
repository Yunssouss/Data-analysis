from deepchem.feat import ConvMolFeaturizer
from deepchem.utils import save_to_disk

smiles = ["CN1CCC(=C2C3=CC=CC=C3C=CC4=CC=CC=C42)CC1"]  # هاد SMILES ديال المركبات اللي جبتي
featurizer = ConvMolFeaturizer()
compounds = featurizer.featurize(smiles)

# حفظ البيانات فـ ملف
save_to_disk(compounds, "compounds_data.pkl")

import deepchem as dc
from deepchem.models import GraphConvModel

# تحميل البيانات ديال tox21
tasks, datasets, transformers = dc.molnet.load_tox21()
train_dataset, valid_dataset, test_dataset = datasets


# بناء موديل GCN
model = GraphConvModel(n_tasks=len(tasks), mode='classification', dropout=0.2)

# تدريب الموديل
model.fit(train_dataset, nb_epoch=50)

metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
evaluation = model.evaluate(test_dataset, [metric], transformers)
print("ROC-AUC Score:", evaluation)

smiles = ["CN1CCC(=C2C3=CC=CC=C3C=CC4=CC=CC=C42)CC1"]  # هاد SMILES ديال المركبات اللي بغيت تجرب
compounds = featurizer.featurize(smiles)

# نديرو التنبؤ
predictions = model.predict_on_batch(compounds)
print("Predictions:", predictions)

