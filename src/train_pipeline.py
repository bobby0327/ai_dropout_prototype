import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Load dataset
df = pd.read_csv('data/sample_students.csv')

X = df.drop(['student_id','enrollment_date','dropout_within_1yr'], axis=1)
y = df['dropout_within_1yr']

numeric_features = ['age','gpa','attendance_pct','assignments_submitted','lms_logins_30d','forum_posts_30d']
categorical_features = ['gender']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

probs = pipeline.predict_proba(X_test)[:,1]
print('ROC-AUC:', roc_auc_score(y_test, probs))
print(classification_report(y_test, pipeline.predict(X_test)))

# Save model
joblib.dump(pipeline, 'models/model_pipeline.joblib')
print("✅ Model saved in models/model_pipeline.joblib")
