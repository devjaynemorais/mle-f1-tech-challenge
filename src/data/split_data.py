

from sklearn.model_selection import train_test_split

def split_data(X, y, meta, seed=42):
    X_train, X_temp, y_train, y_temp, meta_train, meta_temp = train_test_split(
        X, y, meta,
        test_size=0.3,
        stratify=y,
        random_state=seed
    )

    X_val, X_test, y_val, y_test, meta_val, meta_test = train_test_split(
        X_temp, y_temp, meta_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=seed
    )

    return {
        "train": (X_train, y_train, meta_train),
        "val": (X_val, y_val, meta_val),
        "test": (X_test, y_test, meta_test)
    }