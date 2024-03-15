import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import joblib

# Select only the columns present in the original column list
original_columns = (['duration', 'service', 'src_bytes', 'dst_bytes', 'wrong_fragment',
                     'hot', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count', 'attack'])
data = pd.read_csv('C:/Users/sunil/Downloads/KDDTrain+.txt/KDDTrain+.txt', header=None)  # Specify header=None
data.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level']

selected_features = original_columns[:-1]  # Exclude the target column 'attack'
print(data.columns)

# Convert categorical variables to numerical using label encoding
from sklearn.preprocessing import LabelEncoder
categorical_columns = ['service']
label_encoders = {}

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

X = data[selected_features]
y = data['attack']

# Perform feature selection using SelectKBest with chi-squared test
selector = SelectKBest(score_func=chi2, k=10)
X_new = selector.fit_transform(X, y)

# Print the selected features
selected_features = X.columns[selector.get_support(indices=True)]
print("Selected Features:", selected_features)

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Train Random Forest classifier with the selected features
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model
joblib.dump(clf, 'random_forest_model_reduced.pkl')