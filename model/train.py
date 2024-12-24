from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

def train_model():
    # Load Digits dataset from sklearn (similar to MNIST)
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))  # Flatten images into 1D arrays
    y = digits.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the SVM model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    # Save the trained model and scaler
    with open("model/svm_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("model/scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    print("Model training complete and saved.")

if __name__ == "__main__":
    train_model()
