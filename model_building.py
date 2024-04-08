import pickle
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

def preprocessing():    
    with open('dataset.json', 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    df = df.dropna()

    # Encoding both externalStatus and internalStatus
    label_encoder = LabelEncoder()
    df['externalStatus_encoded'] = label_encoder.fit_transform(df['externalStatus'])
    df['internalStatus_encoded'] = label_encoder.fit_transform(df['internalStatus'])

    # Dropping original columns
    df = df.drop(['externalStatus', 'internalStatus'], axis=1)   
    return df, label_encoder

def model_building():
    df, label_encoder = preprocessing()
    X = df[['externalStatus_encoded']]  # Reshaping to 2D array
    Y = df['internalStatus_encoded']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)
    
    # Building and training SVM model
    svm_model = SVC(kernel='linear', C=0.1, gamma=1)
    svm_model.fit(X_train, Y_train)

    # Save the trained model to a file
    with open('svm_model.pkl', 'wb') as file:
        pickle.dump(svm_model, file)

    # Save the label encoder to a file
    with open('label_encoder.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)

    # Making predictions
    Y_pred = svm_model.predict(X_test)
    
    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='weighted')
    recall = recall_score(Y_test, Y_pred, average='weighted')
    
    return accuracy, precision, recall

# Call the model_building function to execute the code and get accuracy, precision, and recall
accuracy, precision, recall = model_building()

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
