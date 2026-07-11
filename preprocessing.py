import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import TEST_SIZE, RANDOM_STATE, SCALER_FILE

def remove_duplicates(data):
   duplicate_count = data.duplicated().sum()
   print(f"Number of duplicate rows: {duplicate_count}")
   if duplicate_count > 0:
         data = data.drop_duplicates().reset_index(drop=True)
         print(f"Removed {duplicate_count} duplicate rows.")
    else:
         print("No duplicate rows found.")
    return data

def split_data(X,y):
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print("Data split completed.")
    print(f"training samples: {X_train.shape[0]}")
    print(f"test samples: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test

def scale_feature(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Feature scaling completed. Scaler saved to {SCALER_FILE}.")
    return X_train_scaled, X_test_scaled
   
def preprocessing_pipeline(data, X, y):
    data = remove_duplicates(data)
    X_train, y_train, X_test, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_feature(X_train, X_test) 
    return data, X_train_scaled, y_train, X_test_scaled, y_test

    
